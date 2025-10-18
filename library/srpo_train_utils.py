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
from contextlib import nullcontext

from . import srpo_reward_models
from . import sdxl_train_util

logger = logging.getLogger(__name__)


def _unpack_unet_output(output):
    """Return the primary tensor from a UNet forward output."""
    if hasattr(output, "sample"):
        return output.sample
    if isinstance(output, (list, tuple)):
        return output[0]
    return output


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
    latents = latents.to(device=device, dtype=weight_dtype)

    if not hasattr(run_sdxl_sample_step, "_srpo_logged_shapes"):
        logger.info(
            "SRPO sample conditioning shapes -- encoder_hidden_states=%s, pooled=%s, add_time_ids=%s",
            tuple(encoder_hidden_states.shape),
            tuple(pooled_prompt_embeds.shape),
            tuple(add_time_ids.shape),
        )
        run_sdxl_sample_step._srpo_logged_shapes = True
    vector_embedding = torch.cat([pooled_prompt_embeds, add_time_ids], dim=-1)

    for i, t in enumerate(timesteps):
        # Expand latents for classifier-free guidance
        latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
        # Keep latent input in model weight dtype to match embeddings (avoids dtype assert in UNet)
        latent_model_input = latent_model_input.to(device=device, dtype=weight_dtype)

        # Predict noise
        with torch.no_grad():
            noise_pred = unet(
                latent_model_input,
                t,
                context=encoder_hidden_states,
                y=vector_embedding,
                return_dict=False,
            )
            noise_pred = _unpack_unet_output(noise_pred)

        if not hasattr(run_sdxl_sample_step, "_srpo_logged_unet_io"):
            logger.info(
                "SRPO sample UNet IO -- latent_in=%s, noise_raw=%s",
                tuple(latent_model_input.shape),
                tuple(noise_pred.shape),
            )
            run_sdxl_sample_step._srpo_logged_unet_io = True
        
        # Perform guidance
        if guidance_scale > 1.0:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        if not hasattr(run_sdxl_sample_step, "_srpo_logged_unet_post"):
            logger.info(
                "SRPO sample post-CFG -- noise_pred=%s, latents=%s",
                tuple(noise_pred.shape),
                tuple(latents.shape),
            )
            run_sdxl_sample_step._srpo_logged_unet_post = True
        
        # Compute previous sample
        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        # Some schedulers return fp32 by default; cast back to the model's weight dtype
        latents = latents.to(device=device, dtype=weight_dtype)
    
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
    # Support percentage-based training window for consistency across different
    # --srpo_timestep_length settings. If provided, map [0,1] -> [0, T-1].
    if getattr(args, 'srpo_train_timestep_pct', None) is not None:
        pct = getattr(args, 'srpo_train_timestep_pct')
        try:
            p0, p1 = float(pct[0]), float(pct[1])
        except Exception:
            p0, p1 = 0.05, 0.25
        p0 = max(0.0, min(1.0, p0))
        p1 = max(0.0, min(1.0, p1))
        if p1 < p0:
            p0, p1 = p1, p0
        lo_idx = int(round(p0 * max(1, timestep_length - 1)))
        hi_idx = int(round(p1 * max(1, timestep_length - 1)))
        train_timestep_range = [lo_idx, hi_idx]
    else:
        train_timestep_range = getattr(args, 'srpo_train_timestep', [5, 25])
    groundtruth_ratio = getattr(args, 'srpo_groundtruth_ratio', 0.9)
    guidance_scale = getattr(args, 'srpo_guidance_scale', 3.5)
    reward_threshold = getattr(args, 'srpo_reward_threshold', 0.7)
    
    # Get custom control words (fall back to defaults if not provided)
    pos_control_words = getattr(args, 'srpo_positive_controls', None)
    neg_control_words = getattr(args, 'srpo_negative_controls', None)
    
    # Discount factors for reward weighting
    discount = torch.linspace(
        discount_pos[0], discount_pos[1], timestep_length, device=device, dtype=weight_dtype
    )
    discount_inversion = torch.linspace(
        discount_inv[0], discount_inv[1], timestep_length, device=device, dtype=weight_dtype
    )
    
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

    if encoder_hidden_states1 is None or encoder_hidden_states2 is None or pooled_prompt_embeds is None:
        raise RuntimeError("SRPO requires text encoder outputs for SDXL but they were missing in the batch.")
    
    # Concatenate hidden states for SDXL
    encoder_hidden_states = torch.cat([encoder_hidden_states1, encoder_hidden_states2], dim=-1)

    if not hasattr(srpo_train_step_sdxl, "_srpo_logged_text_shapes"):
        logger.info(
            "SRPO encodings shapes -- enc1=%s, enc2=%s, pooled=%s, concatenated=%s",
            tuple(encoder_hidden_states1.shape),
            tuple(encoder_hidden_states2.shape),
            tuple(pooled_prompt_embeds.shape),
            tuple(encoder_hidden_states.shape),
        )
        srpo_train_step_sdxl._srpo_logged_text_shapes = True
    
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
    
    # Determine whether we use a reward model (affects downstream execution but rollout is still required
    # to construct x_t by re-noising x0). Keep base conditionings untouched for training call.
    use_reward_model = getattr(args, 'srpo_use_reward_model', True) and reward_model is not None

    # Step 1: Online rollout to get x0 sample (always required to construct x_t)
    latents_x0 = torch.randn((batch_size, 4, height // 8, width // 8), device=device, dtype=weight_dtype)
    # Prepare CFG-expanded conditionings only for the rollout to avoid enlarging tensors used by training UNet call
    enc_for_rollout = encoder_hidden_states
    pool_for_rollout = pooled_prompt_embeds
    timeids_for_rollout = add_time_ids
    if guidance_scale > 1.0:
        enc_for_rollout = torch.cat([enc_for_rollout] * 2)
        pool_for_rollout = torch.cat([pool_for_rollout] * 2)
        timeids_for_rollout = torch.cat([timeids_for_rollout] * 2)

    with torch.no_grad():
        latents_x0 = run_sdxl_sample_step(
            unet=unet,
            scheduler=scheduler,
            latents=latents_x0,
            encoder_hidden_states=enc_for_rollout,
            pooled_prompt_embeds=pool_for_rollout,
            add_time_ids=timeids_for_rollout,
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

    # Shared noise prior ε and common t_start quantities
    latent_shape = (batch_size, 4, height // 8, width // 8)
    eps = torch.randn(latent_shape, device=device, dtype=weight_dtype)

    start_i = mid_timestep  # same for both branches
    t_start_idx = map_idx(start_i)
    alpha_start, sigma_start = _alpha_sigma_from_scheduler(scheduler, t_start_idx, device, weight_dtype)

    # Inject noise at t_start: x_t = alpha_start * x0 + sigma_start * eps
    x_t = alpha_start * latents_x0 + sigma_start * eps
    x_t = x_t.to(device=device, dtype=weight_dtype)

    # Predict noise at t_start once and reuse for both branches
    unet.train()
    t_tensor = torch.tensor([t_start_idx], device=device, dtype=torch.long)
    target_dtype = x_t.dtype
    encoder_hidden_states_local = encoder_hidden_states.to(device=device, dtype=target_dtype)
    pooled_prompt_embeds_local = pooled_prompt_embeds.to(device=device, dtype=target_dtype)
    add_time_ids_local = add_time_ids.to(device=device, dtype=target_dtype)
    vector_embedding_local = torch.cat([pooled_prompt_embeds_local, add_time_ids_local], dim=-1).to(dtype=target_dtype)
    unet_latents = x_t.detach().to(dtype=target_dtype)
    unet_latents.requires_grad_(True)
    noise_pred = unet(
        unet_latents,
        t_tensor,
        context=encoder_hidden_states_local,
        y=vector_embedding_local,
        return_dict=False,
    )
    noise_pred = _unpack_unet_output(noise_pred)

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

    losses = []
    if use_reward_model:
        # Prepare shared VAE context once
        vae.eval()
        vae_dtype = next(vae.parameters()).dtype
        use_cuda_autocast = isinstance(device, str) and device.startswith("cuda") and torch.cuda.is_available()
        target_device = torch.device(device) if isinstance(device, str) else device
        vae_param_device = next(vae.parameters()).device
        if vae_param_device != target_device:
            vae.to(target_device, dtype=vae_dtype)

        # Compute both branch x0 hats, then decode in a single pass to halve VAE overhead
        branch_specs = (
            ("denoise", max(0, mid_timestep - k), discount[mid_timestep]),
            ("inversion", min(timestep_length - 1, mid_timestep + k), discount_inversion[mid_timestep]),
        )
        x0_hats, k_coeffs, names = [], [], []
        for name, end_i, k_coeff in branch_specs:
            t_end_idx = map_idx(end_i)
            alpha_end, sigma_end = _alpha_sigma_from_scheduler(scheduler, t_end_idx, device, weight_dtype)
            dsigma = sigma_end - sigma_start
            x_end = x_t + dsigma * noise_pred
            x0_hat = (x_end - sigma_end * eps) / torch.clamp(alpha_end, min=1e-6)
            x0_hats.append(x0_hat)
            k_coeffs.append(k_coeff)
            names.append(name)

        decode_ctx = torch.autocast("cuda", dtype=vae_dtype) if use_cuda_autocast else nullcontext()
        with decode_ctx:
            dec_in = torch.cat([x / vae.config.scaling_factor for x in x0_hats], dim=0).to(device=target_device, dtype=vae_dtype)
            images_all = vae.decode(dec_in, return_dict=False)[0]
            images_all = (images_all / 2 + 0.5).clamp(0, 1)
        images_all = images_all.to(device=target_device, dtype=torch.float32)

        bs = batch_size
        images_denoise, images_inversion = images_all[:bs], images_all[bs:]
        reward_ctx = torch.autocast("cuda", dtype=torch.float32) if use_cuda_autocast else nullcontext()
        with reward_ctx:
            rewards_denoise = reward_model.srp_cfg(pos_captions, neg_captions, images_denoise, k_coeffs[0])
            rewards_inversion = reward_model.srp_cfg(neg_captions, pos_captions, images_inversion, k_coeffs[1])
        losses.append(F.relu(-rewards_denoise + reward_threshold).mean())
        losses.append(F.relu(-rewards_inversion + reward_threshold).mean())
    else:
        # Preference-weighted MSE w.r.t. rollout x0 for both branches (no VAE/reward model)
        end_i_denoise = max(0, mid_timestep - k)
        end_i_inversion = min(timestep_length - 1, mid_timestep + k)
        # Denoise branch
        alpha_end, sigma_end = _alpha_sigma_from_scheduler(scheduler, map_idx(end_i_denoise), device, weight_dtype)
        x_end = x_t + (sigma_end - sigma_start) * noise_pred
        x0_hat = (x_end - sigma_end * eps) / torch.clamp(alpha_end, min=1e-6)
        weight = (1.0 + discount[mid_timestep]).float()
        losses.append(F.mse_loss(x0_hat.float(), latents_x0.float(), reduction='mean') * weight)
        # Inversion branch
        alpha_end, sigma_end = _alpha_sigma_from_scheduler(scheduler, map_idx(end_i_inversion), device, weight_dtype)
        x_end = x_t + (sigma_end - sigma_start) * noise_pred
        x0_hat = (x_end - sigma_end * eps) / torch.clamp(alpha_end, min=1e-6)
        weight = (1.0 - discount_inversion[mid_timestep]).float()
        losses.append(F.mse_loss(x0_hat.float(), latents_x0.float(), reduction='mean') * weight)

    # Ensure the returned loss is float32 for a stable backward pass under AMP/bfloat16.
    losses = [l.float() for l in losses]
    loss = sum(losses) / len(losses)
    return loss


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
    
    if args.srpo_use_reward_model:
        logger.info(f"SRPO Training enabled with {reward_model_type} reward model")
    else:
        logger.info("SRPO Training enabled")
    logger.info(f"SRPO timestep length: {getattr(args, 'srpo_timestep_length', 100)}")
    logger.info(f"SRPO discount (pos): {getattr(args, 'srpo_discount_pos', [0.1, 0.25])}")
    logger.info(f"SRPO discount (inv): {getattr(args, 'srpo_discount_inv', [0.3, 0.01])}")
