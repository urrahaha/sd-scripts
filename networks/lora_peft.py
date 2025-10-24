"""
PEFT adapter loader for sd-scripts LoRA interface.

Goals
- Allow using PEFT-exported LoRA adapters (adapter_model.safetensors/.bin)
  as a network_module in sd-scripts (e.g. gen_img_diffusers, minimal inference).
- Training flow remains identical to Standard LoRA by delegating to networks.lora.

Usage
- In places where you’d pass `--network_module networks.lora`, you can instead
  pass `--network_module networks.lora_peft` when your `--network_weights`
  points to a PEFT adapter directory or adapter file.

Supported input layouts
- Directory with subfolders: `unet/adapter_model.safetensors` and/or
  `text_encoder/adapter_model.safetensors` (typical PEFT save_pretrained).
- Single-file adapter with consolidated keys (contains `base_model.model.*`).
- `.safetensors` or `.bin` formats.

Notes
- We map PEFT keys like:
  base_model.model.unet.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora_A.weight
  -> lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lora_down.weight

- Alpha values are optional. If not present in the adapter, LoRA alpha defaults to rank in downstream code.
"""

from __future__ import annotations

import os
import json
from typing import Dict, Optional, Tuple, Any
import types

import torch


def _is_safetensors(path: str) -> bool:
    return os.path.splitext(path)[1].lower() == ".safetensors"


def _load_state_dict_any(path: str) -> Dict[str, torch.Tensor]:
    if _is_safetensors(path):
        from safetensors.torch import load_file

        return load_file(path)
    # Fallback to torch.load
    return torch.load(path, map_location="cpu")


def _maybe_load_adapter_config(path: str) -> Optional[dict]:
    cfg_path = os.path.join(path, "adapter_config.json") if os.path.isdir(path) else None
    if cfg_path and os.path.isfile(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def _collect_peft_state_dict(root: str) -> Dict[str, torch.Tensor]:
    """Collect PEFT adapter weights from a root directory or single file.

    Returns a flat state dict whose keys likely begin with `base_model.model.`
    or already with `unet.`/`text_encoder.` prefixes depending on the exporter.
    """
    if os.path.isdir(root):
        # Typical PEFT layout: subfolders per base submodule
        merged: Dict[str, torch.Tensor] = {}
        for sub in ("unet", "text_encoder"):
            cand = os.path.join(root, sub, "adapter_model.safetensors")
            if not os.path.isfile(cand):
                cand = os.path.join(root, sub, "adapter_model.bin")
            if os.path.isfile(cand):
                sd = _load_state_dict_any(cand)
                # If exporter doesn’t include base_model.model.*, add sub prefix
                for k, v in sd.items():
                    if k.startswith("base_model.model."):
                        merged[k] = v
                    else:
                        merged[f"base_model.model.{sub}.{k}"] = v
        # If nothing found via subfolders, try direct file at root
        if not merged:
            for fname in ("adapter_model.safetensors", "adapter_model.bin"):
                p = os.path.join(root, fname)
                if os.path.isfile(p):
                    merged = _load_state_dict_any(p)
                    break
        return merged

    # Single file path
    return _load_state_dict_any(root)


def _strip_prefix(text: str, prefix: str) -> str:
    return text[len(prefix) :] if text.startswith(prefix) else text


def _peft_key_to_kohya_base_key(peft_key: str) -> Optional[Tuple[str, str]]:
    """Return (model_prefix, modulename) where model_prefix in {lora_unet, lora_te}.

    peft_key is a module path like:
      unet.down_blocks.0...attn1.to_q
      text_encoder.text_model.encoder.layers.0....
    We detect top-level and return remaining path.
    """
    k = peft_key
    if k.startswith("unet."):
        return "lora_unet", _strip_prefix(k, "unet.")
    if k.startswith("text_encoder."):
        return "lora_te", _strip_prefix(k, "text_encoder.")
    # Sometimes exporters omit top-level; try best-effort guess
    # Heuristic: CLIP text encoder usually contains "text_model." in path
    if "text_model." in k or k.startswith("model.text_model"):
        return "lora_te", k
    # Default to unet
    return "lora_unet", k


def _convert_peft_sd_to_kohya(peft_sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Map PEFT adapter keys to sd-scripts LoRA keys.

    - base_model.model.<peft_module_path>.lora_A.weight ->
      <kohya_base>_<peft_module_path_with_underscores>.lora_down.weight
    - ...lora_B.weight -> ...lora_up.weight
    - ...alpha (if present) -> ...alpha
    """
    out: Dict[str, torch.Tensor] = {}

    for k, v in peft_sd.items():
        # Normalize prefix
        kk = _strip_prefix(k, "base_model.model.")

        if kk.endswith(".lora_A.weight"):
            mod = kk[: -len(".lora_A.weight")]
            model_prefix, mod_path = _peft_key_to_kohya_base_key(mod)
            kohya_key = f"{model_prefix}_" + mod_path.replace(".", "_") + ".lora_down.weight"
            out[kohya_key] = v
        elif kk.endswith(".lora_B.weight"):
            mod = kk[: -len(".lora_B.weight")]
            model_prefix, mod_path = _peft_key_to_kohya_base_key(mod)
            kohya_key = f"{model_prefix}_" + mod_path.replace(".", "_") + ".lora_up.weight"
            out[kohya_key] = v
        elif kk.endswith(".alpha"):
            mod = kk[: -len(".alpha")]
            model_prefix, mod_path = _peft_key_to_kohya_base_key(mod)
            kohya_key = f"{model_prefix}_" + mod_path.replace(".", "_") + ".alpha"
            out[kohya_key] = v
        else:
            # ignore non-LoRA keys
            continue

    return out


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
    """Training path: just delegate to Standard LoRA implementation.

    This makes the PEFT type behave like Standard for training. Loading from PEFT
    stays supported via create_network_from_weights.
    """
    from . import lora as kohya_lora

    net = kohya_lora.create_network(
        multiplier,
        network_dim,
        network_alpha,
        vae,
        text_encoder,
        unet,
        neuron_dropout=neuron_dropout,
        **kwargs,
    )
    _ensure_scale_weight_norms_support(net)
    return net


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
    """Load a PEFT adapter and construct a sd-scripts LoRA network for it.

    Returns (network, kohya_style_state_dict).
    """
    from . import lora as kohya_lora

    if weights_sd is None:
        if not file:
            raise ValueError("PEFT loader requires a file or weights_sd")
        peft_sd = _collect_peft_state_dict(file)
        if not peft_sd:
            raise ValueError(f"No PEFT adapter weights found in: {file}")
        kohya_sd = _convert_peft_sd_to_kohya(peft_sd)
    else:
        # If caller already provided sd in Kohya format, just use it
        kohya_sd = weights_sd

    # Let the standard LoRA builder derive dims/alphas and instantiate the network
    network, _ = kohya_lora.create_network_from_weights(
        multiplier, None, vae, text_encoder, unet, weights_sd=kohya_sd, for_inference=for_inference, **kwargs
    )
    _ensure_scale_weight_norms_support(network)
    return network, kohya_sd


def _ensure_scale_weight_norms_support(network: torch.nn.Module) -> None:
    """Bind apply_max_norm_regularization to network if missing.

    Mirrors the implementation used by Standard LoRA so `--scale_weight_norms`
    works seamlessly when using PEFT adapters.
    """
    if hasattr(network, "apply_max_norm_regularization"):
        return

    def _apply_max_norm_regularization(self, max_norm_value, device):
        downkeys = []
        upkeys = []
        alphakeys = []
        norms = []
        keys_scaled = 0

        state_dict = self.state_dict()
        for key in list(state_dict.keys()):
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

            # Linear or Conv2d handling like Standard LoRA
            if up.dim() == 4 and down.dim() == 4 and up.shape[2:] == (1, 1) and down.shape[2:] == (1, 1):
                updown = (up.squeeze(3).squeeze(2) @ down.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
            elif up.dim() == 4 and down.dim() == 4 and (up.shape[2:] == (3, 3) or down.shape[2:] == (3, 3)):
                updown = torch.nn.functional.conv2d(down.permute(1, 0, 2, 3), up).permute(1, 0, 2, 3)
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

        mean_norm = (sum(norms) / len(norms)) if norms else 0.0
        max_norm = (max(norms)) if norms else 0.0
        return keys_scaled, mean_norm, max_norm

    try:
        network.apply_max_norm_regularization = types.MethodType(_apply_max_norm_regularization, network)
    except Exception:
        # Best-effort binding; if it fails, the network likely already implements it or
        # cannot be patched at runtime (rare). Training code will handle gracefully.
        pass
