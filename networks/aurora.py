from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple, Type, Union

import torch
from diffusers import AutoencoderKL
from transformers import CLIPTextModel

from library.sdxl_original_unet import SdxlUNet2DConditionModel

from .lora import (
    LoRANetwork,
    LoRAModule,
    LoRAInfModule,
    get_block_dims_and_alphas,
    get_block_lr_weight,
    remove_block_dims_and_alphas,
    convert_diffusers_to_sai_if_needed,
)


class AuroRAModule(LoRAModule):
    def __init__(
        self,
        lora_name,
        org_module: torch.nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        dropout=None,
        rank_dropout=None,
        module_dropout=None,
    ):
        super().__init__(
            lora_name,
            org_module,
            multiplier,
            lora_dim,
            alpha,
            dropout,
            rank_dropout,
            module_dropout,
        )

        if org_module.__class__.__name__ == "Conv2d":
            self.anl_H = torch.nn.Conv2d(self.lora_dim, self.lora_dim, (1, 1), (1, 1), bias=False)
            with torch.no_grad():
                self.anl_H.weight.zero_()
                r = self.lora_dim
                for i in range(r):
                    if i < r:
                        self.anl_H.weight[i, i, 0, 0] = 1.0
        else:
            self.anl_H = torch.nn.Linear(self.lora_dim, self.lora_dim, bias=False)
            with torch.no_grad():
                self.anl_H.weight.copy_(torch.eye(self.lora_dim))

        self.spline_k = 4
        # Centers cover the [-2, 2] support window used for the cardinal cubic B-spline basis.
        self.register_buffer("spline_centers", torch.tensor([-1.5, -0.5, 0.5, 1.5]))
        # scale stretches the knot spacing; default keeps the basis close to paper's [-1, 1] regime.
        self.register_buffer("spline_scale", torch.tensor(1.0))
        if org_module.__class__.__name__ == "Conv2d":
            self.spline_ws = torch.nn.Parameter(torch.empty(self.lora_dim, self.spline_k, 1, 1))
        else:
            self.spline_ws = torch.nn.Parameter(torch.empty(self.lora_dim, self.spline_k))
        torch.nn.init.normal_(self.spline_ws, mean=0.0, std=1e-3)
        self.spline_gate = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        org_forwarded = self.org_forward(x)

        if self.module_dropout is not None and self.training:
            if torch.rand(1) < self.module_dropout:
                return org_forwarded

        lx = self.lora_down(x)

        if self.dropout is not None and self.training:
            lx = torch.nn.functional.dropout(lx, p=self.dropout)

        if self.rank_dropout is not None and self.training:
            mask = torch.rand((lx.size(0), self.lora_dim), device=lx.device) > self.rank_dropout
            if len(lx.size()) == 3:
                mask = mask.unsqueeze(1)
            elif len(lx.size()) == 4:
                mask = mask.unsqueeze(-1).unsqueeze(-1)
            lx = lx * mask
            scale = self.scale * (1.0 / (1.0 - self.rank_dropout))
        else:
            scale = self.scale

        nonlinear_fixed = torch.tanh(self.anl_H(torch.tanh(lx)))
        nonlinear_learned = self._spline_aug(lx)
        lx = nonlinear_fixed + self.spline_gate * nonlinear_learned
        lx = self.lora_up(lx)

        return org_forwarded + lx * self.multiplier * scale

    def _spline_aug(self, z: torch.Tensor) -> torch.Tensor:
        K = self.spline_k
        aug = torch.zeros_like(z)
        for m in range(K):
            c = self.spline_centers[m]
            a = self.spline_scale
            phi = torch.tanh(a * (z - c))
            if z.ndim == 4:
                w = self.spline_ws[:, m]
                if w.ndim > 1:
                    w = w.squeeze()
                w = w.view(1, -1, 1, 1)
            elif z.ndim == 3:
                w = (self.spline_ws[:, m].squeeze()).view(1, 1, -1)
            elif z.ndim == 2:
                w = (self.spline_ws[:, m].squeeze()).view(1, -1)
            else:
                w = (self.spline_ws[:, m].squeeze()).view([1] * (z.ndim - 1) + [-1])
            aug = aug + phi * w
        return aug


class AuroRAInfModule(LoRAInfModule):
    def __init__(
        self,
        lora_name,
        org_module: torch.nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        **kwargs,
    ):
        super().__init__(lora_name, org_module, multiplier, lora_dim, alpha)

        if org_module.__class__.__name__ == "Conv2d":
            self.anl_H = torch.nn.Conv2d(self.lora_dim, self.lora_dim, (1, 1), (1, 1), bias=False)
            with torch.no_grad():
                self.anl_H.weight.zero_()
                r = self.lora_dim
                for i in range(r):
                    if i < r:
                        self.anl_H.weight[i, i, 0, 0] = 1.0
        else:
            self.anl_H = torch.nn.Linear(self.lora_dim, self.lora_dim, bias=False)
            with torch.no_grad():
                self.anl_H.weight.copy_(torch.eye(self.lora_dim))

        self.spline_k = 4
        self.register_buffer("spline_centers", torch.tensor([-1.0, -0.5, 0.5, 1.0]))
        self.register_buffer("spline_scale", torch.tensor(1.5))
        if org_module.__class__.__name__ == "Conv2d":
            self.spline_ws = torch.nn.Parameter(torch.zeros(self.lora_dim, self.spline_k, 1, 1))
        else:
            self.spline_ws = torch.nn.Parameter(torch.zeros(self.lora_dim, self.spline_k))
        self.spline_gate = torch.nn.Parameter(torch.tensor(0.0))

    def anl_forward(self, x):
        z = self.lora_down(x)
        nonlinear_fixed = torch.tanh(self.anl_H(torch.tanh(z)))
        nonlinear_learned = self._spline_aug(z)
        return self.lora_up(nonlinear_fixed + self.spline_gate * nonlinear_learned)

    def default_forward(self, x):
        return self.org_forward(x) + self.anl_forward(x) * self.multiplier * self.scale

    def regional_forward(self, x):
        if "attn2_to_out" in self.lora_name:
            return self.to_out_forward(x)

        if self.network.mask_dic is None:
            return self.default_forward(x)

        lx = self.anl_forward(x) * self.multiplier * self.scale
        mask = self.get_mask_for_x(lx)
        lx = lx * mask

        x = self.org_forward(x)
        x = x + lx

        if "attn2_to_q" in self.lora_name and self.network.is_last_network:
            x = self.postp_to_q(x)

        return x

    def sub_prompt_forward(self, x):
        if x.size()[0] == self.network.batch_size:
            return self.org_forward(x)

        emb_idx = self.network.sub_prompt_index
        if not self.text_encoder:
            emb_idx += self.network.batch_size

        lx = x[emb_idx :: self.network.num_sub_prompts]
        lx = self.anl_forward(lx) * self.multiplier * self.scale

        x = self.org_forward(x)
        x[emb_idx :: self.network.num_sub_prompts] += lx

        return x

    def to_out_forward(self, x):
        if self.network.is_last_network:
            masks = [None] * self.network.num_sub_prompts
            self.network.shared[self.lora_name] = (None, masks)
        else:
            lx, masks = self.network.shared[self.lora_name]

        x1 = x[self.network.batch_size + self.network.sub_prompt_index :: self.network.num_sub_prompts]
        lx1 = self.anl_forward(x1) * self.multiplier * self.scale

        if self.network.is_last_network:
            lx = torch.zeros(
                (self.network.num_sub_prompts * self.network.batch_size, *lx1.size()[1:]), device=lx1.device, dtype=lx1.dtype
            )
            self.network.shared[self.lora_name] = (lx, masks)

        lx[self.network.sub_prompt_index :: self.network.num_sub_prompts] += lx1
        masks[self.network.sub_prompt_index] = self.get_mask_for_x(lx1)

        x = self.org_forward(x)
        if not self.network.is_last_network:
            return x

        lx, masks = self.network.shared.pop(self.lora_name)

        has_real_uncond = x.size()[0] // self.network.batch_size == self.network.num_sub_prompts + 2

        out = torch.zeros((self.network.batch_size * (3 if has_real_uncond else 2), *x.size()[1:]), device=x.device, dtype=x.dtype)
        out[: self.network.batch_size] = x[: self.network.batch_size]
        if has_real_uncond:
            out[-self.network.batch_size :] = x[-self.network.batch_size :]

        for i in range(len(masks)):
            if masks[i] is None:
                masks[i] = torch.zeros_like(masks[0])

        mask = torch.cat(masks)
        mask_sum = torch.sum(mask, dim=0) + 1e-4
        for i in range(self.network.batch_size):
            lx1 = lx[i * self.network.num_sub_prompts : (i + 1) * self.network.num_sub_prompts]
            lx1 = lx1 * mask
            lx1 = torch.sum(lx1, dim=0)

            xi = self.network.batch_size + i * self.network.num_sub_prompts
            x1 = x[xi : xi + self.network.num_sub_prompts]
            x1 = x1 * mask
            x1 = torch.sum(x1, dim=0)
            x1 = x1 / mask_sum

            x1 = x1 + lx1
            out[self.network.batch_size + i] = x1

        return out

    def _anl_on_down_weight(self, down_weight: torch.Tensor, device: torch.device) -> torch.Tensor:
        beta = self.spline_gate.to(torch.float).to(device)
        if len(down_weight.size()) == 2:
            A = down_weight.to(torch.float).to(device)
            H = self.anl_H.weight.to(torch.float).to(device)
            fixed = torch.tanh(H @ torch.tanh(A))
            learned = self._spline_aug_weight(A)
            return fixed + beta * learned
        else:
            A = down_weight.to(torch.float).to(device)
            H = self.anl_H.weight.squeeze(3).squeeze(2).to(torch.float).to(device)
            fixed = torch.tanh(torch.einsum("or,rihw->oihw", H, torch.tanh(A)))
            learned = self._spline_aug_weight(A)
            return fixed + beta * learned

    def _spline_aug(self, z: torch.Tensor) -> torch.Tensor:
        basis_vals = self._evaluate_spline_basis(z)
        aug = torch.zeros_like(z)
        for idx in range(self.spline_k):
            if z.ndim == 4:
                w = self.spline_ws[:, idx].view(1, -1, 1, 1)
            elif z.ndim == 3:
                w = self.spline_ws[:, idx].view(1, 1, -1)
            elif z.ndim == 2:
                w = self.spline_ws[:, idx].view(1, -1)
            else:
                w = self.spline_ws[:, idx].view([1] * (z.ndim - 1) + [-1])
            aug = aug + basis_vals[idx] * w
        return aug

    def _spline_aug_weight(self, A: torch.Tensor) -> torch.Tensor:
        basis_vals = self._evaluate_spline_basis(A)
        aug = torch.zeros_like(A)
        for idx in range(self.spline_k):
            w = self.spline_ws[:, idx]
            if w.ndim > 1:
                w = w.squeeze()
            if len(A.size()) == 2:
                wv = w.view(-1, 1)
            else:
                wv = w.view(-1, 1, 1, 1)
            aug = aug + basis_vals[idx] * wv
        return aug

    def _cardinal_cubic_bspline(self, u: torch.Tensor) -> torch.Tensor:
        abs_u = torch.abs(u)
        result = torch.zeros_like(u)
        mask1 = abs_u < 1
        mask2 = (abs_u >= 1) & (abs_u < 2)
        if mask1.any():
            result = result + (((4 - 6 * abs_u**2 + 3 * abs_u**3) / 6) * mask1)
        if mask2.any():
            result = result + ((((2 - abs_u) ** 3) / 6) * mask2)
        return result

    def _evaluate_spline_basis(self, z: torch.Tensor) -> List[torch.Tensor]:
        scale = torch.clamp(self.spline_scale, min=1e-6)
        basis = []
        for center in self.spline_centers:
            u = (z - center) / scale
            basis.append(self._cardinal_cubic_bspline(u))
        return basis

    def merge_to(self, sd, dtype, device):
        up_weight = sd["lora_up.weight"].to(torch.float).to(device)
        down_weight = sd["lora_down.weight"].to(torch.float).to(device)

        org_sd = self.org_module.state_dict()
        weight = org_sd["weight"].to(torch.float)

        A_tilde = self._anl_on_down_weight(down_weight, device)

        if len(weight.size()) == 2:
            upd = up_weight @ A_tilde
            weight = weight + self.multiplier * upd * self.scale
        elif down_weight.size()[2:4] == (1, 1):
            up2 = up_weight.squeeze(3).squeeze(2)
            upd = (up2 @ A_tilde.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
            weight = weight + self.multiplier * upd * self.scale
        else:
            conved = torch.nn.functional.conv2d(A_tilde.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3)
            weight = weight + self.multiplier * conved * self.scale

        org_sd["weight"] = weight.to(dtype)
        self.org_module.load_state_dict(org_sd)

    def get_weight(self, multiplier=None):
        if multiplier is None:
            multiplier = self.multiplier

        up_weight = self.lora_up.weight.to(torch.float)
        down_weight = self.lora_down.weight.to(torch.float)
        A_tilde = self._anl_on_down_weight(down_weight, up_weight.device)

        if len(down_weight.size()) == 2:
            upd = up_weight @ A_tilde
        elif down_weight.size()[2:4] == (1, 1):
            up2 = up_weight.squeeze(3).squeeze(2)
            upd = (up2 @ A_tilde.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
        else:
            upd = torch.nn.functional.conv2d(A_tilde.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3)

        return multiplier * upd * self.scale


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

    conv_dim = kwargs.get("conv_dim", None)
    conv_alpha = kwargs.get("conv_alpha", None)
    if conv_dim is not None:
        conv_dim = int(conv_dim)
        if conv_alpha is None:
            conv_alpha = 1.0
        else:
            conv_alpha = float(conv_alpha)

    block_dims = kwargs.get("block_dims", None)
    block_lr_weight = get_block_lr_weight(
        is_sdxl,
        kwargs.get("down_lr_weight", None),
        kwargs.get("mid_lr_weight", None),
        kwargs.get("up_lr_weight", None),
        float(kwargs.get("block_lr_zero_threshold", 0.0)),
    )

    if block_dims is not None or block_lr_weight is not None:
        block_alphas = kwargs.get("block_alphas", None)
        conv_block_dims = kwargs.get("conv_block_dims", None)
        conv_block_alphas = kwargs.get("conv_block_alphas", None)

        block_dims, block_alphas, conv_block_dims, conv_block_alphas = get_block_dims_and_alphas(
            is_sdxl, block_dims, block_alphas, network_dim, network_alpha, conv_block_dims, conv_block_alphas, conv_dim, conv_alpha
        )

        block_dims, block_alphas, conv_block_dims, conv_block_alphas = remove_block_dims_and_alphas(
            is_sdxl, block_dims, block_alphas, conv_block_dims, conv_block_alphas, block_lr_weight
        )
    else:
        block_alphas = None
        conv_block_dims = None
        conv_block_alphas = None

    rank_dropout = kwargs.get("rank_dropout", None)
    if rank_dropout is not None:
        rank_dropout = float(rank_dropout)
    module_dropout = kwargs.get("module_dropout", None)
    if module_dropout is not None:
        module_dropout = float(module_dropout)

    network = LoRANetwork(
        text_encoder,
        unet,
        multiplier=multiplier,
        lora_dim=network_dim,
        alpha=network_alpha,
        dropout=neuron_dropout,
        rank_dropout=rank_dropout,
        module_dropout=module_dropout,
        conv_lora_dim=conv_dim,
        conv_alpha=conv_alpha,
        block_dims=block_dims,
        block_alphas=block_alphas,
        conv_block_dims=conv_block_dims,
        conv_block_alphas=conv_block_alphas,
        module_class=AuroRAModule,
        varbose=True,
        is_sdxl=is_sdxl,
    )

    # Apply optional spline settings from kwargs to all created modules
    spline_gate = kwargs.get("spline_gate", None)
    spline_scale = kwargs.get("spline_scale", None)
    spline_centers = kwargs.get("spline_centers", None)

    def _parse_centers(val, K):
        if val is None:
            return None
        if isinstance(val, (list, tuple)):
            arr = [float(x) for x in val]
        else:
            try:
                arr = [float(x.strip()) for x in str(val).split(",") if x.strip() != ""]
            except Exception:
                return None
        if len(arr) < 1:
            return None
        # If length mismatches, truncate/pad to K
        if len(arr) < K:
            arr = arr + [arr[-1]] * (K - len(arr))
        elif len(arr) > K:
            arr = arr[:K]
        return torch.tensor(arr)

    if spline_gate is not None or spline_scale is not None or spline_centers is not None:
        loras = []
        loras.extend(getattr(network, "text_encoder_loras", []))
        loras.extend(getattr(network, "unet_loras", []))
        for m in loras:
            if hasattr(m, "spline_gate"):
                if spline_gate is not None:
                    try:
                        m.spline_gate.data = torch.tensor(float(spline_gate))
                    except Exception:
                        pass
                if spline_scale is not None and hasattr(m, "spline_scale"):
                    try:
                        m.spline_scale.data = torch.tensor(float(spline_scale))
                    except Exception:
                        pass
                if spline_centers is not None and hasattr(m, "spline_centers"):
                    centers_t = _parse_centers(spline_centers, getattr(m, "spline_k", 4))
                    if centers_t is not None:
                        m.spline_centers.data = centers_t

    loraplus_lr_ratio = kwargs.get("loraplus_lr_ratio", None)
    loraplus_unet_lr_ratio = kwargs.get("loraplus_unet_lr_ratio", None)
    loraplus_text_encoder_lr_ratio = kwargs.get("loraplus_text_encoder_lr_ratio", None)
    loraplus_lr_ratio = float(loraplus_lr_ratio) if loraplus_lr_ratio is not None else None
    loraplus_unet_lr_ratio = float(loraplus_unet_lr_ratio) if loraplus_unet_lr_ratio is not None else None
    loraplus_text_encoder_lr_ratio = (
        float(loraplus_text_encoder_lr_ratio) if loraplus_text_encoder_lr_ratio is not None else None
    )
    if loraplus_lr_ratio is not None or loraplus_unet_lr_ratio is not None or loraplus_text_encoder_lr_ratio is not None:
        network.set_loraplus_lr_ratio(loraplus_lr_ratio, loraplus_unet_lr_ratio, loraplus_text_encoder_lr_ratio)

    if block_lr_weight is not None:
        network.set_block_lr_weight(block_lr_weight)

    return network


def create_network_from_weights(
    multiplier,
    file,
    vae,
    text_encoder,
    unet,
    weights_sd=None,
    for_inference=False,
    **kwargs,
):
    is_sdxl = unet is not None and issubclass(unet.__class__, SdxlUNet2DConditionModel)

    if weights_sd is None:
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")

    if is_sdxl:
        convert_diffusers_to_sai_if_needed(weights_sd)

    modules_dim = {}
    modules_alpha = {}
    for key, value in weights_sd.items():
        if "." not in key:
            continue
        lora_name = key.split(".")[0]
        if "alpha" in key:
            modules_alpha[lora_name] = value
        elif "lora_down" in key:
            dim = value.size()[0]
            modules_dim[lora_name] = dim

    for key in list(modules_dim.keys()):
        if key not in modules_alpha:
            modules_alpha[key] = modules_dim[key]

    module_class: Type[LoRAModule] = AuroRAInfModule if for_inference else AuroRAModule

    network = LoRANetwork(
        text_encoder,
        unet,
        multiplier=multiplier,
        modules_dim=modules_dim,
        modules_alpha=modules_alpha,
        module_class=module_class,
        is_sdxl=is_sdxl,
    )

    block_lr_weight = get_block_lr_weight(
        is_sdxl,
        kwargs.get("down_lr_weight", None),
        kwargs.get("mid_lr_weight", None),
        kwargs.get("up_lr_weight", None),
        float(kwargs.get("block_lr_zero_threshold", 0.0)),
    )
    if block_lr_weight is not None:
        network.set_block_lr_weight(block_lr_weight)

    return network, weights_sd
