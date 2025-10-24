from __future__ import annotations

import math
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
    get_block_index,
    convert_diffusers_to_sai_if_needed,
)


class NLoraModule(LoRAModule):
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
            self.lora_n = torch.nn.Conv2d(self.lora_dim, self.lora_dim, (1, 1), (1, 1), bias=False)
        else:
            self.lora_n = torch.nn.Linear(self.lora_dim, self.lora_dim, bias=False)

        torch.nn.init.kaiming_uniform_(self.lora_n.weight, a=math.sqrt(5))

        try:
            w = self.org_module.weight.detach()
            r = int(self.lora_dim)
            if w.dim() == 2:
                if w.size(0) >= r and w.size(1) >= r:
                    with torch.no_grad():
                        self.lora_down.weight.copy_(w[:r, :])
                        self.lora_up.weight.copy_(w[:, :r])
                        if isinstance(self.lora_n, torch.nn.Linear):
                            self.lora_n.weight.copy_(w[:r, :r])
            elif w.dim() == 4 and w.shape[2:] == (1, 1):
                if w.size(0) >= r and w.size(1) >= r:
                    with torch.no_grad():
                        self.lora_down.weight.copy_(w[:r, :, :, :])
                        self.lora_up.weight.copy_(w[:, :r, :, :])
                        if isinstance(self.lora_n, torch.nn.Conv2d) and self.lora_n.weight.shape[2:] == (1, 1):
                            self.lora_n.weight.copy_(w[:r, :r, :, :])
        except Exception:
            pass

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

        lx = self.lora_n(lx)
        lx = self.lora_up(lx)

        return org_forwarded + lx * self.multiplier * scale


class NLoraInfModule(LoRAInfModule):
    def default_forward(self, x):
        return self.org_forward(x) + self.lora_up(self.lora_n(self.lora_down(x))) * self.multiplier * self.scale

    def regional_forward(self, x):
        if "attn2_to_out" in self.lora_name:
            return self.to_out_forward(x)

        if self.network.mask_dic is None:
            return self.default_forward(x)

        lx = self.lora_up(self.lora_n(self.lora_down(x))) * self.multiplier * self.scale
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
        lx = self.lora_up(self.lora_n(self.lora_down(lx))) * self.multiplier * self.scale

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
        lx1 = self.lora_up(self.lora_n(self.lora_down(x1))) * self.multiplier * self.scale

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

    def merge_to(self, sd, dtype, device):
        up_weight = sd["lora_up.weight"].to(torch.float).to(device)
        down_weight = sd["lora_down.weight"].to(torch.float).to(device)
        n_weight = sd.get("lora_n.weight", None)
        if n_weight is not None:
            n_weight = n_weight.to(torch.float).to(device)

        org_sd = self.org_module.state_dict()
        weight = org_sd["weight"].to(torch.float)

        if len(weight.size()) == 2:
            if n_weight is None:
                upd = up_weight @ down_weight
            else:
                upd = up_weight @ n_weight @ down_weight
            weight = weight + self.multiplier * upd * self.scale
        elif down_weight.size()[2:4] == (1, 1):
            up2 = up_weight.squeeze(3).squeeze(2)
            if n_weight is not None:
                n2 = n_weight.squeeze(3).squeeze(2)
                up2 = up2 @ n2
            upd = (up2 @ down_weight.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
            weight = weight + self.multiplier * upd * self.scale
        else:
            up2 = up_weight
            if n_weight is not None:
                up2 = (up_weight.squeeze(3).squeeze(2) @ n_weight.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
            conved = torch.nn.functional.conv2d(down_weight.permute(1, 0, 2, 3), up2).permute(1, 0, 2, 3)
            weight = weight + self.multiplier * conved * self.scale

        org_sd["weight"] = weight.to(dtype)
        self.org_module.load_state_dict(org_sd)

    def get_weight(self, multiplier=None):
        if multiplier is None:
            multiplier = self.multiplier

        up_weight = self.lora_up.weight.to(torch.float)
        down_weight = self.lora_down.weight.to(torch.float)
        n_weight = getattr(self, "lora_n", None)
        n_weight = n_weight.weight.to(torch.float) if n_weight is not None else None

        if len(down_weight.size()) == 2:
            upd = up_weight @ down_weight if n_weight is None else up_weight @ n_weight @ down_weight
        elif down_weight.size()[2:4] == (1, 1):
            up2 = up_weight.squeeze(3).squeeze(2)
            if n_weight is not None:
                up2 = up2 @ n_weight.squeeze(3).squeeze(2)
            upd = (up2 @ down_weight.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
        else:
            up2 = up_weight
            if n_weight is not None:
                up2 = (up_weight.squeeze(3).squeeze(2) @ n_weight.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
            upd = torch.nn.functional.conv2d(down_weight.permute(1, 0, 2, 3), up2).permute(1, 0, 2, 3)

        return multiplier * upd * self.scale


class NLoraNetwork(LoRANetwork):
    def apply_max_norm_regularization(self, max_norm_value, device):
        downkeys = []
        upkeys = []
        nkeys = []
        alphakeys = []
        norms = []
        keys_scaled = 0

        state_dict = self.state_dict()
        for key in state_dict.keys():
            if "lora_down" in key and "weight" in key:
                downkeys.append(key)
                upkeys.append(key.replace("lora_down", "lora_up"))
                n_key = key.replace("lora_down", "lora_n")
                nkeys.append(n_key if n_key in state_dict else None)
                alphakeys.append(key.replace("lora_down.weight", "alpha"))

        for i in range(len(downkeys)):
            down = state_dict[downkeys[i]].to(device)
            up = state_dict[upkeys[i]].to(device)
            alpha = state_dict[alphakeys[i]].to(device)
            n = state_dict[nkeys[i]].to(device) if nkeys[i] is not None else None
            dim = down.shape[0]
            scale = alpha / dim

            if up.shape[2:] == (1, 1) and down.shape[2:] == (1, 1):
                up2 = up.squeeze(2).squeeze(2)
                if n is not None:
                    up2 = up2 @ n.squeeze(2).squeeze(2)
                updown = (up2 @ down.squeeze(2).squeeze(2)).unsqueeze(2).unsqueeze(3)
            # conv2d kxk (currently k=3) case: down is kxk and up is 1x1
            elif up.shape[2:] == (3, 3) or down.shape[2:] == (3, 3):
                up2 = up
                if n is not None:
                    up2 = (up.squeeze(3).squeeze(2) @ n.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
                updown = torch.nn.functional.conv2d(down.permute(1, 0, 2, 3), up2).permute(1, 0, 2, 3)
            else:
                updown = up @ (n @ down if n is not None else down)

            updown *= scale

            norm = updown.norm().clamp(min=max_norm_value / 2)
            desired = torch.clamp(norm, max=max_norm_value)
            ratio = desired.cpu() / norm.cpu()
            sqrt_ratio = ratio**0.5
            if ratio != 1:
                keys_scaled += 1
                state_dict[upkeys[i]] *= sqrt_ratio
                state_dict[downkeys[i]] *= sqrt_ratio
                if nkeys[i] is not None:
                    state_dict[nkeys[i]] *= sqrt_ratio
            scalednorm = updown.norm() * ratio
            norms.append(scalednorm.item())

        return keys_scaled, sum(norms) / len(norms), max(norms)


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

    # rsLoRA alpha remapping (alpha -> alpha * sqrt(rank))
    def _truthy(v: object) -> bool:
        return str(v).lower() in {"1", "true", "yes", "y", "on"}

    use_rslora = _truthy(kwargs.get("rslora", False) or kwargs.get("use_rslora", False))
    if use_rslora:
        if network_dim is not None and network_alpha is not None:
            network_alpha = float(network_alpha) * math.sqrt(int(network_dim))
        if conv_dim is not None and conv_alpha is not None:
            conv_alpha = float(conv_alpha) * math.sqrt(int(conv_dim))
        if block_alphas is not None and block_dims is not None:
            block_alphas = [float(a) * math.sqrt(int(d)) for a, d in zip(block_alphas, block_dims)]
        if conv_block_alphas is not None and conv_block_dims is not None:
            conv_block_alphas = [float(a) * math.sqrt(int(d)) for a, d in zip(conv_block_alphas, conv_block_dims)]
        # No special casing for NLora's extra N matrix: scale is still alpha/r in base LoRA math

    network = NLoraNetwork(
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
        module_class=NLoraModule,
        varbose=True,
        is_sdxl=is_sdxl,
    )

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

    module_class: Type[LoRAModule] = NLoraInfModule if for_inference else NLoraModule

    network = NLoraNetwork(
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
