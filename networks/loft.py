# LoFT network module (SDXL-focused)
# Implements alternating low-rank updates and projected gradient scaling hooks
# Paper: "LoFT: Low-Rank Adaptation That Behaves Like Full Fine-Tuning" (arXiv:2505.21289)

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Tuple, Type, Union

import torch
from diffusers import AutoencoderKL
from transformers import CLIPTextModel

from library.utils import setup_logging
from library.sdxl_original_unet import SdxlUNet2DConditionModel

setup_logging()
import logging

logger = logging.getLogger(__name__)


class LoFTModule(torch.nn.Module):
    """
    LoFT low-rank adapter module. Follows the LoRA structure but adds:
    - Alternating updates between U (up) and V (down) each training step
    - Projected gradient scaling using (V^T V)^{-1} or (U^T U)^{-1} via per-parameter hooks

    Notes:
    - U is represented by `lora_up.weight` with shape [out_dim, r] (or [out_dim, r, 1, 1] for Conv2d 1x1)
    - V^T is represented by `lora_down.weight` with shape [r, in_dim] (or [r, in_dim, kH, kW] for Conv2d)
    """

    def __init__(
        self,
        loft_name: str,
        org_module: torch.nn.Module,
        multiplier: float = 1.0,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        self.loft_name = loft_name
        self.multiplier = multiplier
        self.rank = rank
        self.dropout = dropout

        # infer io dims
        if org_module.__class__.__name__ == "Conv2d":
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels
            kernel_size = org_module.kernel_size
            stride = org_module.stride
            padding = org_module.padding
            # V^T : in->rank with source kernel size; U : rank->out with 1x1
            self.lora_down = torch.nn.Conv2d(in_dim, self.rank, kernel_size, stride, padding, bias=False)
            self.lora_up = torch.nn.Conv2d(self.rank, out_dim, (1, 1), (1, 1), bias=False)
        else:
            in_dim = org_module.in_features
            out_dim = org_module.out_features
            self.lora_down = torch.nn.Linear(in_dim, self.rank, bias=False)
            self.lora_up = torch.nn.Linear(self.rank, out_dim, bias=False)

        # LoFT uses same scaling as LoRA for inference compatibility
        if isinstance(alpha, torch.Tensor):
            alpha = alpha.detach().float().item()
        alpha = self.rank if alpha is None or alpha == 0 else float(alpha)
        self.scale = alpha / self.rank
        self.register_buffer("alpha", torch.tensor(alpha))

        # init like LoRA
        torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_up.weight)

        # tag parameters for LoFT-aware optimizers (role + peer)
        try:
            self.lora_up.weight._loft_role = "U"
            self.lora_up.weight._loft_peer = self.lora_down.weight
            self.lora_down.weight._loft_role = "V"
            self.lora_down.weight._loft_peer = self.lora_up.weight
        except Exception:
            pass

        self.org_module = org_module  # kept until apply_to()
        self._u_hook = None
        self._v_hook = None
        self.enabled = True

    # --- helpers ---
    @staticmethod
    def _to_2d(w: torch.Tensor) -> torch.Tensor:
        return w.flatten(1) if w.dim() > 2 else w

    def _clear_hooks(self):
        if self._u_hook is not None:
            try:
                self._u_hook.remove()
            except Exception:
                pass
            self._u_hook = None
        if self._v_hook is not None:
            try:
                self._v_hook.remove()
            except Exception:
                pass
            self._v_hook = None

    # --- public API ---
    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def set_update_direction(self, update_u: bool):
        """
        Alternate which factor is trainable and install projected gradient hooks.
        update_u=True  => train U (lora_up); freeze V (lora_down)
        update_u=False => train V; freeze U
        """
        # remove previous hooks
        self._clear_hooks()

        self.lora_up.weight.requires_grad_(update_u)
        self.lora_down.weight.requires_grad_(not update_u)

        eps = 1e-6
        if update_u:
            # scale grad_U := grad_U @ (V^T V + eps I)^{-1}
            def _u_grad_hook(grad: torch.Tensor) -> torch.Tensor:
                Vt = self._to_2d(self.lora_down.weight)  # [r, in*kw*kh]
                Vt32 = Vt.float()
                gram32 = Vt32 @ Vt32.t()  # [r, r] == V^T V
                gram32 = gram32 + eps * torch.eye(gram32.shape[0], device=gram32.device, dtype=torch.float32)
                gram_inv32 = torch.linalg.inv(gram32)
                if grad.dim() > 2:
                    g2d = grad.flatten(1).float()
                    g2d = g2d @ gram_inv32
                    return g2d.to(dtype=grad.dtype).view_as(grad)
                else:
                    return (grad.float() @ gram_inv32).to(dtype=grad.dtype)

            self._u_hook = self.lora_up.weight.register_hook(_u_grad_hook)
        else:
            # scale grad_{V^T} := (U^T U + eps I)^{-1} @ grad_{V^T}
            def _v_grad_hook(grad: torch.Tensor) -> torch.Tensor:
                U = self._to_2d(self.lora_up.weight)  # [out, r]
                U32 = U.float()
                gram32 = U32.t() @ U32  # [r, r] == U^T U
                gram32 = gram32 + eps * torch.eye(gram32.shape[0], device=gram32.device, dtype=torch.float32)
                gram_inv32 = torch.linalg.inv(gram32)
                if grad.dim() > 2:
                    g2d = grad.flatten(1).float()
                    g2d = gram_inv32 @ g2d
                    return g2d.to(dtype=grad.dtype).view_as(grad)
                else:
                    return (gram_inv32 @ grad.float()).to(dtype=grad.dtype)

            self._v_hook = self.lora_down.weight.register_hook(_v_grad_hook)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        org_forwarded = self.org_forward(x)
        if not self.enabled:
            return org_forwarded

        lx = self.lora_down(x)
        if self.dropout is not None and self.training:
            lx = torch.nn.functional.dropout(lx, p=self.dropout)
        lx = self.lora_up(lx)
        return org_forwarded + lx * self.multiplier * self.scale


class LoFTNetwork(torch.nn.Module):
    """
    SDXL-focused LoFT network with LoRA-compatible I/O and training hooks used by train_network.py
    """

    UNET_TARGET_REPLACE_MODULE = ["Transformer2DModel"]
    UNET_TARGET_REPLACE_MODULE_CONV2D_3X3 = ["ResnetBlock2D", "Downsample2D", "Upsample2D"]
    TEXT_ENCODER_TARGET_REPLACE_MODULE = ["CLIPAttention", "CLIPSdpaAttention", "CLIPMLP"]

    LOFT_PREFIX_UNET = "loft_unet"
    LOFT_PREFIX_TEXT_ENCODER = "loft_te"
    LOFT_PREFIX_TEXT_ENCODER1 = "loft_te1"  # SDXL uses two text encoders
    LOFT_PREFIX_TEXT_ENCODER2 = "loft_te2"

    def __init__(
        self,
        text_encoder: Union[List[CLIPTextModel], CLIPTextModel],
        unet,
        multiplier: float = 1.0,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: Optional[float] = None,
        modules_rank: Optional[Dict[str, int]] = None,
        modules_alpha: Optional[Dict[str, float]] = None,
        verbose: bool = True,
        is_sdxl: bool = True,
    ) -> None:
        super().__init__()
        self.multiplier = multiplier
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.is_sdxl = is_sdxl
        self._update_u = False  # will be set on first step in prepare_grad_etc

        def create_modules(
            is_unet: bool,
            text_encoder_idx: Optional[int],
            root: torch.nn.Module,
            target_types: List[str],
        ) -> Tuple[List[LoFTModule], List[str]]:
            prefix = (
                self.LOFT_PREFIX_UNET
                if is_unet
                else (
                    self.LOFT_PREFIX_TEXT_ENCODER
                    if text_encoder_idx is None
                    else (self.LOFT_PREFIX_TEXT_ENCODER1 if text_encoder_idx == 1 else self.LOFT_PREFIX_TEXT_ENCODER2)
                )
            )
            lofts: List[LoFTModule] = []
            skipped: List[str] = []
            for name, module in root.named_modules():
                if module.__class__.__name__ in target_types:
                    for child_name, child_module in module.named_modules():
                        is_linear = child_module.__class__.__name__ == "Linear"
                        is_conv2d = child_module.__class__.__name__ == "Conv2d"
                        is_conv2d_1x1 = is_conv2d and getattr(child_module, "kernel_size", None) == (1, 1)

                        if not (is_linear or is_conv2d):
                            continue

                        # For UNet, support Conv2d 3x3 by placing the low-rank into down (k x k) and up (1x1)
                        if is_conv2d and (not is_conv2d_1x1):
                            # allowed (down: kxk, up: 1x1)
                            pass

                        loft_name = (prefix + "." + name + "." + child_name).replace(".", "_")

                        dim = None
                        alp = None
                        if modules_rank is not None and loft_name in modules_rank:
                            dim = int(modules_rank[loft_name])
                            alp = float(modules_alpha[loft_name]) if modules_alpha and loft_name in modules_alpha else float(self.alpha)
                        else:
                            dim = self.rank
                            alp = self.alpha

                        if dim is None or dim == 0:
                            skipped.append(loft_name)
                            continue

                        l = LoFTModule(loft_name, child_module, self.multiplier, dim, alp, dropout=self.dropout)
                        lofts.append(l)
            return lofts, skipped

        text_encoders = text_encoder if isinstance(text_encoder, list) else [text_encoder]

        # Text encoders (SDXL can have two)
        self.text_encoder_lofts: List[LoFTModule] = []
        for i, te in enumerate(text_encoders):
            idx = i + 1 if len(text_encoders) > 1 else None
            lofts, skipped = create_modules(False, idx, te, self.TEXT_ENCODER_TARGET_REPLACE_MODULE)
            self.text_encoder_lofts.extend(lofts)
            if verbose:
                logger.info(f"create LoFT for Text Encoder{(' ' + str(idx)) if idx else ''}: {len(lofts)} modules")
                if skipped:
                    logger.info(f"skipped {len(skipped)} modules")

        # U-Net
        target_modules = list(self.UNET_TARGET_REPLACE_MODULE)
        target_modules += self.UNET_TARGET_REPLACE_MODULE_CONV2D_3X3  # allow conv 3x3 path
        self.unet_lofts, skipped_un = create_modules(True, None, unet, target_modules)
        if verbose:
            logger.info(f"create LoFT for U-Net: {len(self.unet_lofts)} modules")
            if skipped_un:
                logger.info(f"skipped {len(skipped_un)} modules")

    # --- public interface consumed by train_network.py ---
    def set_multiplier(self, multiplier: float):
        self.multiplier = multiplier
        for m in self.text_encoder_lofts + self.unet_lofts:
            m.multiplier = multiplier

    def load_weights(self, file: str):
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")
        info = self.load_state_dict(weights_sd, strict=False)
        return info

    def apply_to(self, text_encoder, unet, apply_text_encoder=True, apply_unet=True):
        if apply_text_encoder:
            logger.info(f"enable LoFT for text encoder: {len(self.text_encoder_lofts)} modules")
        else:
            self.text_encoder_lofts = []

        if apply_unet:
            logger.info(f"enable LoFT for U-Net: {len(self.unet_lofts)} modules")
        else:
            self.unet_lofts = []

        for m in self.text_encoder_lofts + self.unet_lofts:
            m.apply_to()
            self.add_module(m.loft_name, m)

    def is_mergeable(self):
        # same behavior as LoRA modules
        return True

    def prepare_optimizer_params(self, text_encoder_lr, unet_lr, default_lr):
        self.requires_grad_(True)
        all_params = []
        lr_desc = []

        def add_group(mods: List[LoFTModule], lr: Optional[float], desc_prefix: str):
            if not mods:
                return
            params = []
            for m in mods:
                # always register both U and V; requires_grad toggled per step
                params.append(m.lora_up.weight)
                params.append(m.lora_down.weight)
            group = {"params": params}
            if lr is not None:
                group["lr"] = lr
            else:
                group["lr"] = default_lr
            all_params.append(group)
            lr_desc.append(desc_prefix)

        add_group(self.text_encoder_lofts, text_encoder_lr, "textencoder")
        add_group(self.unet_lofts, unet_lr, "unet")
        return all_params, lr_desc

    def enable_gradient_checkpointing(self):
        # not supported; kept for API compatibility
        pass

    def prepare_grad_etc(self, text_encoder, unet):
        # initial direction: update V first (match paper alt updates)
        self._update_u = False
        for m in self.text_encoder_lofts + self.unet_lofts:
            m.set_update_direction(self._update_u)
        self.train()

    def on_epoch_start(self, text_encoder, unet):
        self.train()

    def on_step_start(self, text_encoder, unet):
        # alternate update direction each trainer step
        self._update_u = not self._update_u
        for m in self.text_encoder_lofts + self.unet_lofts:
            m.set_update_direction(self._update_u)

    def get_trainable_params(self):
        return self.parameters()

    def save_weights(self, file: str, dtype: Optional[torch.dtype], metadata: Optional[Dict[str, str]]):
        if metadata is not None and len(metadata) == 0:
            metadata = None

        state_dict = self.state_dict()
        if dtype is not None:
            for k in list(state_dict.keys()):
                v = state_dict[k]
                state_dict[k] = v.detach().clone().to("cpu").to(dtype)

        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import save_file
            from library import train_util

            if metadata is None:
                metadata = {}
            model_hash, legacy_hash = train_util.precalculate_safetensors_hashes(state_dict, metadata)
            metadata["sshs_model_hash"] = model_hash
            metadata["sshs_legacy_hash"] = legacy_hash
            save_file(state_dict, file, metadata)
        else:
            torch.save(state_dict, file)

    # Optional: used when args.scale_weight_norms is set
    def apply_max_norm_regularization(self, max_norm_value, device):
        downkeys = []
        upkeys = []
        alphakeys = []
        norms = []
        keys_scaled = 0

        state_dict = self.state_dict()
        for key in state_dict.keys():
            if "lora_down" in key and "weight" in key:
                downkeys.append(key)
                upkeys.append(key.replace("lora_down", "lora_up"))
                alphakeys.append(key.replace("lora_down.weight", "alpha"))

        for i in range(len(downkeys)):
            down = state_dict[downkeys[i]].to(device)
            up = state_dict[upkeys[i]].to(device)
            alpha = state_dict[alphakeys[i]].to(device)
            dim = down.shape[0]
            scale = alpha / dim

            if up.dim() == 4 and down.dim() == 4 and up.shape[2:] == (1, 1) and down.shape[2:] == down.shape[2:]:
                updown = (up.squeeze(3).squeeze(2) @ down.flatten(1)).unsqueeze(2).unsqueeze(3)
            elif up.dim() == 4 or down.dim() == 4:
                # general conv conv: compute as 2D
                up2d = up.flatten(1)
                down2d = down.flatten(1)
                updown = up2d @ down2d
            else:
                updown = up @ down

            updown *= scale

            norm = updown.norm().clamp(min=max_norm_value / 2)
            desired = torch.clamp(norm, max=max_norm_value)
            ratio = desired.cpu() / norm.cpu()
            sqrt_ratio = ratio ** 0.5
            if ratio != 1:
                keys_scaled += 1
                state_dict[upkeys[i]] *= sqrt_ratio
                state_dict[downkeys[i]] *= sqrt_ratio
            scalednorm = updown.norm() * ratio
            norms.append(scalednorm.item())

        return keys_scaled, (sum(norms) / len(norms) if norms else 0.0), (max(norms) if norms else 0.0)


# --- factory functions expected by train_network.py ---

def create_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae: AutoencoderKL,
    text_encoder: Union[CLIPTextModel, List[CLIPTextModel]],
    unet,
    neuron_dropout: Optional[float] = None,
    **kwargs,
):
    is_sdxl = unet is not None and issubclass(unet.__class__, SdxlUNet2DConditionModel)

    if network_dim is None:
        network_dim = 4
    if network_alpha is None:
        network_alpha = 1.0

    network = LoFTNetwork(
        text_encoder,
        unet,
        multiplier=multiplier,
        rank=int(network_dim),
        alpha=float(network_alpha),
        dropout=neuron_dropout,
        modules_rank=None,
        modules_alpha=None,
        verbose=True,
        is_sdxl=is_sdxl,
    )
    return network


def create_network_from_weights(multiplier, file, vae, text_encoder, unet, weights_sd=None, for_inference=False, **kwargs):
    is_sdxl = unet is not None and issubclass(unet.__class__, SdxlUNet2DConditionModel)

    if weights_sd is None:
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")

    modules_rank: Dict[str, int] = {}
    modules_alpha: Dict[str, float] = {}
    for key, value in weights_sd.items():
        if "." not in key:
            continue
        name = key.split(".")[0]
        if "alpha" in key:
            modules_alpha[name] = float(value)
        elif "lora_down" in key:
            dim = value.size()[0]
            modules_rank[name] = int(dim)

    # fallback: alpha==rank when missing
    for k, v in modules_rank.items():
        if k not in modules_alpha:
            modules_alpha[k] = float(v)

    network = LoFTNetwork(
        text_encoder,
        unet,
        multiplier=multiplier,
        rank=4,  # base, not used when modules_rank provided
        alpha=1.0,
        modules_rank=modules_rank,
        modules_alpha=modules_alpha,
        verbose=True,
        is_sdxl=is_sdxl,
    )
    return network, weights_sd
