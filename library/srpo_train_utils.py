"""
SRPO (Style Reward Preference Optimization) Training Utilities for SDXL
Based on: https://github.com/Tencent-Hunyuan/SRPO

Implements direct alignment of diffusion trajectory with reward models.
Adds Direct-Align closed-form one-step recovery for SDXL.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple
from diffusers.image_processor import VaeImageProcessor
from tqdm import tqdm
import logging

from . import srpo_reward_models
from . import sdxl_train_util

logger = logging.getLogger(__name__)


def prepare_latent_image_ids_sdxl(batch_size, height, width, device, dtype):
    """Prepare latent image IDs for SDXL (not needed for standard SDXL, kept for compatibility)"""
    # SDXL uses standard latent format, no special IDs needed
    return None


def sdxl_euler_step(
    model_output: torch.Tensor,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    prev_timestep: torch.Tensor,
) -> torch.Tensor:
    """
    Single Euler step for SDXL diffusion
    x_{t-1} = x_t + (t - t-1) * model_output
    """
    dt = prev_timestep - timestep
    prev_sample = latents + dt.view(-1, 1, 1, 1) * model_output
    return prev_sample


def _alpha_sigma_from_scheduler(scheduler, t_index: int, device, dtype):
    """Get alpha_t and sigma_t from scheduler.alphas_cumprod at integer index."""
    ac = scheduler.alphas_cumprod
    if not torch.is_tensor(ac):
        ac = torch.tensor(ac, device=device)
    ac = ac.to(device=device, dtype=dtype)
    t_index = int(max(0, min(int(t_index), ac.shape[0] - 1)))
    alpha_cum = ac[t_index]
    alpha_t = torch.sqrt(alpha_cum)
    sigma_t = torch.sqrt(torch.clamp(1.0 - alpha_cum, min=0.0))
    return alpha_t, sigma_t


def run_sdxl_sample_step(
    unet,
    scheduler,
    latents,
    encoder_hidden_states,
    pooled_prompt_embeds,
    add_time_ids,
    guidance_scale,
    num_inference_steps,
    start_step=0,
    end_step=None,
    device="cuda",
    weight_dtype=torch.float32,
):
    """
    Run SDXL sampling steps for online rollout
    
    Args:
        unet: SDXL UNet model
        scheduler: Diffusion scheduler
        latents: Initial latent tensor
        encoder_hidden_states: Text encoder outputs
        pooled_prompt_embeds: Pooled text embeddings
        add_time_ids: Additional time embeddings for SDXL
        guidance_scale: CFG scale
        num_inference_steps: Total inference steps
        start_step: Starting step index
        end_step: Ending step index (None = run to completion)
        device: torch device
        weight_dtype: Weight dtype
    
    Returns:
        Final latents after sampling
    """
    if end_step is None:
        end_step = num_inference_steps
    
    # Set timesteps
    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps[start_step:end_step]
    
    encoder_hidden_states = encoder_hidden_states.to(device=device, dtype=weight_dtype)
    pooled_prompt_embeds = pooled_prompt_embeds.to(device=device, dtype=weight_dtype)
    add_time_ids = add_time_ids.to(device=device, dtype=weight_dtype)
    vector_embedding = torch.cat([pooled_prompt_embeds, add_time_ids], dim=-1)

    for i, t in enumerate(timesteps):
        # Expand latents for classifier-free guidance
        latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents

        # Predict noise
        with torch.no_grad():
            noise_pred = unet(
                latent_model_input,
                t,
                encoder_hidden_states=encoder_hidden_states,
                y=vector_embedding,
                return_dict=False,
            )[0]
        
        # Perform guidance
        if guidance_scale > 1.0:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        # Compute previous sample
        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
    
    return latents


def srpo_train_step_sdxl(
    args,
    unet,
    vae,
    text_encoder1,
    text_encoder2,
    reward_model: Optional[any],
    scheduler,
    batch,
    global_step,
    device="cuda",
    weight_dtype=torch.float32,
    precomputed_encoder_hidden_states1: Optional[torch.Tensor] = None,
    precomputed_encoder_hidden_states2: Optional[torch.Tensor] = None,
    precomputed_pooled_prompt_embeds: Optional[torch.Tensor] = None,
):
    """
    Single SRPO training step for SDXL using Direct-Align closed-form.

    Returns a differentiable scalar loss tensor (do not call backward here).
    """
    # Get SRPO parameters
    timestep_length = getattr(args, 'srpo_timestep_length', 100)
    discount_pos = getattr(args, 'srpo_discount_pos', [0.1, 0.25])
    discount_inv = getattr(args, 'srpo_discount_inv', [0.3, 0.01])
    train_timestep_range = getattr(args, 'srpo_train_timestep', [5, 25])
    groundtruth_ratio = getattr(args, 'srpo_groundtruth_ratio', 0.9)
    guidance_scale = getattr(args, 'srpo_guidance_scale', 3.5)
    reward_threshold = getattr(args, 'srpo_reward_threshold', 0.7)
    
    # Get custom control words (fall back to defaults if not provided)
    pos_control_words = getattr(args, 'srpo_positive_controls', None)
    neg_control_words = getattr(args, 'srpo_negative_controls', None)
    
    # Discount factors for reward weighting
    discount = torch.linspace(discount_pos[0], discount_pos[1], timestep_length).to(device)
    discount_inversion = torch.linspace(discount_inv[0], discount_inv[1], timestep_length).to(device)
    
    # Determine batch size and resolution
    if "target_sizes_hw" in batch and batch["target_sizes_hw"] is not None:
        h0, w0 = batch["target_sizes_hw"][0].tolist()
        height, width = int(h0), int(w0)
    elif "original_sizes_hw" in batch and batch["original_sizes_hw"] is not None:
        h0, w0 = batch["original_sizes_hw"][0].tolist()
        height, width = int(h0), int(w0)
    else:
        height, width = 1024, 1024
    if "input_ids" in batch and batch["input_ids"] is not None:
        batch_size = batch["input_ids"].shape[0]
    elif "latents" in batch and batch["latents"] is not None:
        batch_size = batch["latents"].shape[0]
    else:
        batch_size = 1
    
    # Text embeddings (prefer precomputed if provided)
    encoder_hidden_states1 = precomputed_encoder_hidden_states1 if precomputed_encoder_hidden_states1 is not None else batch.get("text_encoder_outputs1")
    encoder_hidden_states2 = precomputed_encoder_hidden_states2 if precomputed_encoder_hidden_states2 is not None else batch.get("text_encoder_outputs2")
    pooled_prompt_embeds = precomputed_pooled_prompt_embeds if precomputed_pooled_prompt_embeds is not None else batch.get("text_encoder_pool2")
    
    # Concatenate hidden states for SDXL
    encoder_hidden_states = torch.cat([encoder_hidden_states1, encoder_hidden_states2], dim=-1)
    
    # Get add_time_ids for SDXL
    add_time_ids = batch.get("add_time_ids")

    # Ensure ADM vector matches SDXL expectations (1536 dims from size embeddings)
    if add_time_ids is not None and add_time_ids.shape[-1] != 1536:
        add_time_ids = None

    if add_time_ids is None:
        # Build size embeddings from batch metadata to obtain the 1536-dim vector
        orig_size = batch.get("original_sizes_hw")
        crop_size = batch.get("crop_top_lefts")
        target_size = batch.get("target_sizes_hw")

        if orig_size is None:
            orig_size = torch.tensor([[height, width]], device=device, dtype=torch.float32)
        else:
            orig_size = orig_size.to(device=device, dtype=torch.float32)

        if crop_size is None:
            crop_size = torch.zeros((batch_size, 2), device=device, dtype=torch.float32)
        else:
            crop_size = crop_size.to(device=device, dtype=torch.float32)

        if target_size is None:
            target_size = torch.tensor([[height, width]], device=device, dtype=torch.float32)
        else:
            target_size = target_size.to(device=device, dtype=torch.float32)

        if orig_size.shape[0] != batch_size:
            orig_size = orig_size.repeat(batch_size, 1)
        if crop_size.shape[0] != batch_size:
            crop_size = crop_size.repeat(batch_size, 1)
        if target_size.shape[0] != batch_size:
            target_size = target_size.repeat(batch_size, 1)

        add_time_ids = sdxl_train_util.get_size_embeddings(orig_size, crop_size, target_size, device=device)

    add_time_ids = add_time_ids.to(device=device, dtype=weight_dtype)
    
    # Expand for CFG if needed
    if guidance_scale > 1.0:
        encoder_hidden_states = torch.cat([encoder_hidden_states] * 2)
        pooled_prompt_embeds = torch.cat([pooled_prompt_embeds] * 2)
        add_time_ids = torch.cat([add_time_ids] * 2)
    
    # Step 1: Online rollout to get x0 sample
    latents_x0 = torch.randn((batch_size, 4, height // 8, width // 8), device=device, dtype=weight_dtype)
    with torch.no_grad():
        latents_x0 = run_sdxl_sample_step(
            unet=unet,
            scheduler=scheduler,
            latents=latents_x0,
            encoder_hidden_states=encoder_hidden_states,
            pooled_prompt_embeds=pooled_prompt_embeds,
            add_time_ids=add_time_ids,
            guidance_scale=guidance_scale,
            num_inference_steps=timestep_length,
            device=device,
            weight_dtype=weight_dtype,
        )
    
    # Add control words for style preference
    captions = batch.get("captions", [""] * batch_size)
    
    # Use custom control words if provided, otherwise fall back to defaults
    if pos_control_words and len(pos_control_words) > 0:
        pos_control = pos_control_words[global_step % len(pos_control_words)]
    else:
        pos_control = srpo_reward_models.get_random_realism_adjective(global_step)
    
    if neg_control_words and len(neg_control_words) > 0:
        neg_control = neg_control_words[global_step % len(neg_control_words)]
    else:
        neg_control = srpo_reward_models.get_random_cg_oily_adjective(global_step)
    
    pos_captions = [f"{pos_control}. {cap}" for cap in captions]
    neg_captions = [f"{neg_control}. {cap}" for cap in captions]
    
    # Choose timestep within configured range
    import random
    lo, hi = int(train_timestep_range[0]), int(train_timestep_range[1])
    lo = max(1, min(lo, timestep_length - 2))
    hi = max(lo + 1, min(hi, timestep_length - 1))
    mid_timestep = random.randint(lo, hi)
    k = int((1.0 - groundtruth_ratio) * timestep_length) + 1
    k = max(1, min(min(timestep_length - 1 - mid_timestep, k), mid_timestep))

    # Map SRPO index [0..T-1] to scheduler index [0..num_train_timesteps-1]
    num_train_t = int(getattr(scheduler.config, 'num_train_timesteps', 1000))
    def map_idx(i: int) -> int:
        return int(round(i / float(max(1, timestep_length - 1)) * (num_train_t - 1)))

    discount = torch.linspace(discount_pos[0], discount_pos[1], timestep_length, device=device, dtype=weight_dtype)
    discount_inversion = torch.linspace(discount_inv[0], discount_inv[1], timestep_length, device=device, dtype=weight_dtype)

    # Captions
    captions = batch.get("captions", [""] * batch_size)
    pos_control_words = getattr(args, 'srpo_positive_controls', None)
    neg_control_words = getattr(args, 'srpo_negative_controls', None)
    if pos_control_words and len(pos_control_words) > 0:
        pos_control = pos_control_words[global_step % len(pos_control_words)]
    else:
        pos_control = srpo_reward_models.get_random_realism_adjective(global_step)
    if neg_control_words and len(neg_control_words) > 0:
        neg_control = neg_control_words[global_step % len(neg_control_words)]
    else:
        neg_control = srpo_reward_models.get_random_cg_oily_adjective(global_step)
    pos_captions = [f"{pos_control}. {cap}" for cap in captions]
    neg_captions = [f"{neg_control}. {cap}" for cap in captions]

    # Shared noise prior ε
    eps = torch.randn_like(latents_x0)

    losses = []
    for branch in ("denoise", "inversion"):
        if branch == "denoise":
            start_i = mid_timestep
            end_i = max(0, mid_timestep - k)
            k_coeff = discount[mid_timestep]
        else:
            start_i = mid_timestep
            end_i = min(timestep_length - 1, mid_timestep + k)
            k_coeff = discount_inversion[mid_timestep]

        t_start_idx = map_idx(start_i)
        t_end_idx = map_idx(end_i)
        alpha_start, sigma_start = _alpha_sigma_from_scheduler(scheduler, t_start_idx, device, weight_dtype)
        alpha_end, sigma_end = _alpha_sigma_from_scheduler(scheduler, t_end_idx, device, weight_dtype)

        # Inject noise at t_start: x_t = alpha_start * x0 + sigma_start * eps
        x_t = alpha_start * latents_x0 + sigma_start * eps

        # Predict noise at t_start
        unet.train()
        t_tensor = torch.tensor([t_start_idx], device=device, dtype=torch.long)
        encoder_hidden_states_local = encoder_hidden_states[:batch_size].to(device=device, dtype=weight_dtype)
        pooled_prompt_embeds_local = pooled_prompt_embeds[:batch_size].to(device=device, dtype=weight_dtype)
        add_time_ids_local = add_time_ids[:batch_size].to(device=device, dtype=weight_dtype)
        vector_embedding_local = torch.cat([pooled_prompt_embeds_local, add_time_ids_local], dim=-1)
        noise_pred = unet(
            x_t.detach().requires_grad_(True),
            t_tensor,
            encoder_hidden_states=encoder_hidden_states_local,
            y=vector_embedding_local,
            return_dict=False,
        )[0]

        # If requested, convert Diff2Flow-style v-parameterized output to epsilon using scheduler alphas
        # This aligns the single-step update with diffusion semantics when UNet outputs v.
        if getattr(args, 'srpo_use_diff2flow', False):
            # Determine parameterization from args override or scheduler config
            d2f_param = getattr(args, 'srpo_d2f_param', None)
            pred_type = getattr(getattr(scheduler, 'config', object()), 'prediction_type', 'epsilon')
            # Heuristic: prefer explicit arg; else infer from scheduler config
            use_v = (d2f_param == 'v') or (d2f_param is None and isinstance(pred_type, str) and pred_type.lower().startswith('v'))
            if use_v:
                # Convert v -> eps: eps = sqrt(alpha_cum) * v + sqrt(1 - alpha_cum) * x_t
                # alpha_start is sqrt(alpha_cum), sigma_start is sqrt(1 - alpha_cum)
                noise_pred = (alpha_start * noise_pred) + (sigma_start * x_t)

        # One-step move: x_end = x_t + (sigma_end - sigma_start) * noise_pred
        dsigma = sigma_end - sigma_start
        x_end = x_t + dsigma * noise_pred

        # Closed-form recovery at t_end
        x0_hat = (x_end - sigma_end * eps) / torch.clamp(alpha_end, min=1e-6)

        use_reward_model = getattr(args, 'srpo_use_reward_model', True) and reward_model is not None
        if use_reward_model:
            vae.eval()
            with torch.autocast("cuda", dtype=weight_dtype):
                dec_in = x0_hat / vae.config.scaling_factor
                images = vae.decode(dec_in, return_dict=False)[0]
                images = (images / 2 + 0.5).clamp(0, 1)
            with torch.amp.autocast('cuda'):
                if branch == "denoise":
                    rewards = reward_model.srp_cfg(pos_captions, neg_captions, images, k_coeff)
                else:
                    rewards = reward_model.srp_cfg(neg_captions, pos_captions, images, k_coeff)
            loss_branch = F.relu(-rewards + reward_threshold).mean()
        else:
            # Preference-weighted MSE w.r.t. rollout x0
            if branch == "denoise":
                weight = 1.0 + k_coeff
            else:
                weight = 1.0 - k_coeff
            loss_branch = F.mse_loss(x0_hat, latents_x0, reduction='mean') * weight

        losses.append(loss_branch)

    return sum(losses) / len(losses)


def should_use_srpo_training(args) -> bool:
    """Check if SRPO training is enabled"""
    return getattr(args, 'srpo_enable', False)


def validate_srpo_args(args):
    """Validate SRPO arguments"""
    if not should_use_srpo_training(args):
        return
    
    # Check required parameters
    reward_model_type = getattr(args, 'srpo_reward_model', 'HPS')
    if reward_model_type not in ['HPS', 'PickScore', 'CLIP']:
        raise ValueError(f"Invalid SRPO reward model: {reward_model_type}")
    
    # Warn about incompatibilities
    if args.cache_latents or args.cache_latents_to_disk:
        logger.warning("SRPO training requires online generation. Disabling latent caching.")
        args.cache_latents = False
        args.cache_latents_to_disk = False
    
    if getattr(args, 'cache_text_encoder_outputs', False):
        logger.warning("SRPO training works better without cached text encoder outputs for dynamic prompts.")
    
    logger.info(f"SRPO Training enabled with {reward_model_type} reward model")
    logger.info(f"SRPO timestep length: {getattr(args, 'srpo_timestep_length', 100)}")
    logger.info(f"SRPO discount (pos): {getattr(args, 'srpo_discount_pos', [0.1, 0.25])}")
    logger.info(f"SRPO discount (inv): {getattr(args, 'srpo_discount_inv', [0.3, 0.01])}")
