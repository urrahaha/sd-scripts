"""
Generic PEFT training backend for sd-scripts.

This module enables training with any PEFT tuner by:
- Building a PEFT config (from a base64-encoded JSON passed via --network_args),
- Injecting adapters into UNet and/or Text Encoder(s) in-place, preserving the
  original objects used by sd-scripts’ training loop,
- Exposing the adapter parameters to the optimizer and gradient-reduction path,
- Saving/loading adapter weights as a single-file safetensors or torch file with
  keys under `base_model.model.<component>.*` so they can be reloaded or merged.

CLI usage (via GUI already wired):
  --network_module networks.peft_generic \
  --network_args peft_tuner=Lora peft_cfg_b64=<base64-json>

Notes
- Users must provide correct `target_modules` etc. appropriate for their base
  model; this backend does not auto-detect them.
- For SDXL two-text-encoder setups, we fan out to both if requested.
"""

from __future__ import annotations

import base64
import json
import os
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import torch
import torch.nn as nn

# sd-scripts logging helper
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


def _is_safetensors(path: str) -> bool:
    return os.path.splitext(path)[1].lower() == ".safetensors"


def _load_any(path: str) -> Dict[str, torch.Tensor]:
    if _is_safetensors(path):
        from safetensors.torch import load_file

        return load_file(path)
    return torch.load(path, map_location="cpu")


def _save_any(state: Mapping[str, torch.Tensor], path: str, metadata: Optional[Dict[str, str]] = None):
    if _is_safetensors(path):
        from safetensors.torch import save_file

        # metadata can be None
        save_file(dict(state), path, metadata)
    else:
        torch.save(dict(state), path)


def _get_text_encoders(te: Any) -> List[nn.Module]:
    if te is None:
        return []
    return te if isinstance(te, list) else [te]


class PeftNetwork(nn.Module):
    def __init__(self, tuner: str, cfg_dict: Dict[str, Any]):
        super().__init__()
        self.tuner = tuner
        self.cfg_dict = dict(cfg_dict)

        # Sanitize incompatible combinations early (GUI should already do this, but be defensive)
        try:
            tm = self.cfg_dict.get("target_modules")
            if isinstance(tm, str):
                # When using shorthand string (e.g. "all-linear"), PEFT disallows layers_to_transform/layers_pattern
                self.cfg_dict.pop("layers_to_transform", None)
                self.cfg_dict.pop("layers_pattern", None)
        except Exception:
            pass

        # placeholders filled by apply_to
        self._wrappers_te: List[Any] = []
        self._wrappers_unet: List[Any] = []
        self._trainable_params: List[nn.Parameter] = []

    # sd-scripts API -------------------------------------------------------
    def apply_to(self, text_encoder, unet, train_text_encoder: bool = True, train_unet: bool = True):
        try:
            import peft
            from peft import get_peft_model
        except Exception as e:
            raise RuntimeError(f"PEFT is required for PEFT training backend: {e}")

        cfg_cls_name = f"{self.tuner}Config" if not self.tuner.endswith("Config") else self.tuner
        cfg_cls = getattr(peft, cfg_cls_name, None)
        if cfg_cls is None:
            raise ValueError(f"Unknown PEFT tuner: {self.tuner}")

        # Log the config we will instantiate for transparency
        try:
            pretty_json = json.dumps(self.cfg_dict, indent=2, sort_keys=True)
        except Exception:
            pretty_json = str(self.cfg_dict)
        logger.info("[PEFT] Tuner=%s using config JSON:\n%s", self.tuner, pretty_json)

        def _wrap_and_patch(model: nn.Module) -> Any:
            # Instantiate a fresh config per model so that internal expansion (e.g., 'all-linear')
            # is done against the correct module set and does not bleed between TE and UNet.
            peft_cfg_local = cfg_cls(**self.cfg_dict)
            # Give PEFT a stable name_or_path to avoid noisy warnings
            try:
                if getattr(model, "name_or_path", None) in (None, ""):
                    setattr(model, "name_or_path", self.cfg_dict.get("base_model_name_or_path") or "")
            except Exception:
                pass
            wrapper = get_peft_model(model, peft_cfg_local)
            # Do NOT attach wrapper as a submodule of model; that creates a cycle and breaks .to()/._apply().
            return wrapper

        # Text encoders can be a list (SDXL) or single
        if train_text_encoder:
            for te in _get_text_encoders(text_encoder):
                try:
                    w = _wrap_and_patch(te)
                    self._wrappers_te.append(w)
                except Exception as e:
                    logger.warning("PEFT wrapping text encoder failed: %s", e)
        if train_unet and unet is not None:
            try:
                w = _wrap_and_patch(unet)
                self._wrappers_unet.append(w)
            except Exception as e:
                logger.warning("PEFT wrapping unet failed: %s", e)

        # Collect trainable adapter params for optimizer and manual all_reduce
        self._trainable_params = []
        for w in self._wrappers_te + self._wrappers_unet:
            for p in w.parameters():
                if p.requires_grad:
                    self._trainable_params.append(p)

        if not self._trainable_params:
            logger.warning("No trainable PEFT parameters discovered. Check your target_modules/config.")

    def prepare_optimizer_params(self, text_encoder_lr: Optional[float], unet_lr: Optional[float], default_lr: Optional[float] = None):
        # Build param groups split by component to respect separate LRs
        def _params_of(ws: Iterable[Any]) -> List[nn.Parameter]:
            ps: List[nn.Parameter] = []
            for w in ws:
                for n, p in w.named_parameters():
                    if p.requires_grad:
                        ps.append(p)
            return ps

        groups: List[Dict[str, Any]] = []
        te_params = _params_of(self._wrappers_te)
        unet_params = _params_of(self._wrappers_unet)

        if te_params:
            lr = (text_encoder_lr if text_encoder_lr not in (None, 0) else default_lr) or 0.0
            if lr:
                groups.append({"params": te_params, "lr": lr})
        if unet_params:
            lr = (unet_lr if unet_lr not in (None, 0) else default_lr) or 0.0
            if lr:
                groups.append({"params": unet_params, "lr": lr})

        # Fallback single group if none assigned an LR
        if not groups and self._trainable_params:
            lr = default_lr or 0.0
            groups.append({"params": list(self._trainable_params), "lr": lr})

        # Return groups and lr_descriptions (optional second value)
        return groups, ["textencoder" if te_params else None, "unet" if unet_params else None]

    def prepare_optimizer_params_with_multiple_te_lrs(self, text_encoder_lrs: Optional[List[float]], unet_lr: Optional[float], default_lr: Optional[float] = None):
        # For simplicity, use the first LR if provided; extend later if needed
        te_lr = None
        if isinstance(text_encoder_lrs, list) and text_encoder_lrs:
            te_lr = text_encoder_lrs[0]
        return self.prepare_optimizer_params(te_lr, unet_lr, default_lr)

    def enable_gradient_checkpointing(self):
        # Let base models handle checkpointing as usual; nothing required here
        pass

    def get_trainable_params(self) -> Iterable[nn.Parameter]:
        return self._trainable_params

    # Provide parameters() so manual all_reduce sees our adapter tensors
    def parameters(self, recurse: bool = True) -> Iterable[nn.Parameter]:  # type: ignore[override]
        return iter(self._trainable_params)

    # sd-scripts training loop API -----------------------------------------
    def prepare_grad_etc(self, text_encoder, unet):
        # Our .parameters() returns only adapter parameters, so this toggles grads only for adapters
        self.requires_grad_(True)

    def on_epoch_start(self, text_encoder, unet):
        # Ensure adapters are in train mode each epoch
        self.train()

    # Loading/saving --------------------------------------------------------
    def load_weights(self, file: str) -> Mapping[str, Any]:
        try:
            from peft import set_peft_model_state_dict
        except Exception as e:
            raise RuntimeError(f"PEFT is required for loading adapters: {e}")

        sd = _load_any(file)
        te_wrappers = self._wrappers_te
        unet_wrappers = self._wrappers_unet

        def _apply(prefix: str, wrappers: List[Any]):
            # Extract and strip prefix `base_model.model.<prefix>.`
            pfx = f"base_model.model.{prefix}."
            local = {k[len(pfx) :]: v for k, v in sd.items() if k.startswith(pfx)}
            if not local:
                return
            for w in wrappers:
                set_peft_model_state_dict(w, local, strict=False)

        _apply("unet", unet_wrappers)
        # text encoders may be multiple but we store under a common prefix "text_encoder"
        _apply("text_encoder", te_wrappers)
        return {"loaded_keys": list(sd.keys())}

    def save_weights(self, file: str, dtype: Optional[torch.dtype], metadata: Optional[Dict[str, str]]):
        try:
            from peft import get_peft_model_state_dict
        except Exception as e:
            raise RuntimeError(f"PEFT is required for saving adapters: {e}")

        out: Dict[str, torch.Tensor] = {}

        def _collect(prefix: str, wrappers: List[Any]):
            for w in wrappers:
                local = get_peft_model_state_dict(w)
                for k, v in local.items():
                    t = v.detach().clone()
                    if dtype is not None:
                        t = t.to(dtype)
                    out[f"base_model.model.{prefix}.{k}"] = t.cpu()

        _collect("unet", self._wrappers_unet)
        _collect("text_encoder", self._wrappers_te)

        _save_any(out, file, metadata)

    # Norm regularization --------------------------------------------------
    def apply_max_norm_regularization(self, max_norm_value: float, device: torch.device):
        """Scale LoRA adapter weights to enforce a maximum norm.

        Mirrors Standard LoRA behavior so `--scale_weight_norms` works with PEFT.
        Returns (keys_scaled, mean_norm, max_norm).
        """
        try:
            from peft import get_peft_model_state_dict, set_peft_model_state_dict
        except Exception as e:
            raise RuntimeError(f"PEFT is required for adapter norm scaling: {e}")

        def _collect_pairs(local: Mapping[str, torch.Tensor]):
            pairs = []  # list of (key_A, key_B, key_alpha or None)
            # Accept both patterns: '.lora_A.weight' and '.lora_A.default.weight'
            for k in list(local.keys()):
                if ".lora_A." in k and k.endswith("weight"):
                    stem, tail = k.rsplit(".lora_A.", 1)  # stem is module path before lora_A
                    b_key = f"{stem}.lora_B.{tail}"
                    a_key = k
                    # alpha usually at the same stem with '.alpha' (without trailing pieces)
                    alpha_key = f"{stem}.alpha"
                    if b_key in local:
                        pairs.append((a_key, b_key, alpha_key if alpha_key in local else None))
            return pairs

        def _compute_updown(up: torch.Tensor, down: torch.Tensor) -> torch.Tensor:
            # Handle Linear and Conv variants similarly to Standard LoRA
            if up.dim() == 4 and down.dim() == 4 and up.shape[2:] == (1, 1) and down.shape[2:] == (1, 1):
                return (up.squeeze(3).squeeze(2) @ down.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
            if up.dim() == 4 and down.dim() == 4 and (up.shape[2:] == (3, 3) or down.shape[2:] == (3, 3)):
                return torch.nn.functional.conv2d(down.permute(1, 0, 2, 3), up).permute(1, 0, 2, 3)
            return up @ down

        keys_scaled = 0
        norms: List[float] = []

        # Process each wrapped model independently
        for w in self._wrappers_te + self._wrappers_unet:
            if w is None:
                continue
            local = dict(get_peft_model_state_dict(w))
            if not local:
                continue
            pairs = _collect_pairs(local)
            if not pairs:
                continue

            for a_key, b_key, alpha_key in pairs:
                down = local[a_key].to(device)
                up = local[b_key].to(device)
                # default scale = 1 if alpha missing
                dim = down.shape[0] if down.dim() >= 2 else max(down.shape[0], 1)
                if alpha_key is not None:
                    alpha = local[alpha_key].to(device)
                    scale = alpha / dim
                else:
                    scale = torch.tensor(1.0, device=device, dtype=down.dtype)

                updown = _compute_updown(up, down)
                updown *= scale

                norm = updown.norm().clamp(min=max_norm_value / 2)
                desired = torch.clamp(norm, max=max_norm_value)
                ratio = (desired / norm).detach()
                sqrt_ratio = ratio**0.5

                if ratio.item() != 1.0:
                    keys_scaled += 1
                    # write back scaled weights to CPU for set_peft_model_state_dict
                    local[b_key] = (up * sqrt_ratio).to(local[b_key].dtype).cpu()
                    local[a_key] = (down * sqrt_ratio).to(local[a_key].dtype).cpu()
                    if alpha_key is not None:
                        # alpha stays unchanged
                        local[alpha_key] = local[alpha_key].cpu()
                else:
                    # Keep originals (ensure CPU for setter)
                    local[b_key] = up.to(local[b_key].dtype).cpu()
                    local[a_key] = down.to(local[a_key].dtype).cpu()
                    if alpha_key is not None:
                        local[alpha_key] = local[alpha_key].cpu()

                scalednorm = updown.norm() * ratio
                norms.append(float(scalednorm.item()))

            # Apply updated adapter weights back to the wrapper
            set_peft_model_state_dict(w, local, strict=False)

        mean_norm = (sum(norms) / len(norms)) if norms else 0.0
        max_norm = (max(norms)) if norms else 0.0
        return keys_scaled, mean_norm, max_norm


def create_network(
    multiplier: float,
    network_dim: int,
    network_alpha: float,
    vae,
    text_encoder,
    unet,
    neuron_dropout: Optional[float] = None,
    **kwargs: Any,
):
    # Expect `peft_tuner` and `peft_cfg_b64` in kwargs
    tuner = kwargs.get("peft_tuner")
    cfg_b64 = kwargs.get("peft_cfg_b64")
    if not tuner:
        raise ValueError("PEFT backend requires 'peft_tuner' in --network_args")
    if not cfg_b64:
        raise ValueError("PEFT backend requires 'peft_cfg_b64' (base64-encoded JSON config)")

    try:
        cfg_dict = json.loads(base64.urlsafe_b64decode(cfg_b64.encode("ascii")).decode("utf-8"))
    except Exception as e:
        raise ValueError(f"Invalid peft_cfg_b64: {e}")

    return PeftNetwork(tuner, cfg_dict)


def create_network_from_weights(
    multiplier: float,
    file: Optional[str],
    vae,
    text_encoder,
    unet,
    weights_sd: Optional[Dict[str, torch.Tensor]] = None,
    for_inference: bool = False,
    **kwargs: Any,
):
    # Create empty network first, weights loaded after apply_to during training
    nw = create_network(1.0, 0, 0, vae, text_encoder, unet, **kwargs)
    if file:
        # We cannot load until wrappers exist, so return the raw SD for later
        return nw, _load_any(file)
    else:
        return nw, (weights_sd or {})
