"""
SRPO (Style Reward Preference Optimization) Training Utilities for SDXL
Based on: https://github.com/Tencent-Hunyuan/SRPO

Implements direct alignment of diffusion trajectory with reward models.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple
from diffusers.image_processor import VaeImageProcessor
from tqdm import tqdm
import logging

from . import srpo_reward_models

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
    
    for i, t in enumerate(timesteps):
        # Expand latents for classifier-free guidance
        latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
        
        # Predict noise
        with torch.no_grad():
            noise_pred = unet(
                latent_model_input,
                t,
                encoder_hidden_states=encoder_hidden_states,
                added_cond_kwargs={
                    "text_embeds": pooled_prompt_embeds,
                    "time_ids": add_time_ids
                },
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
):
    """
    Single SRPO training step for SDXL
    
    Implements the Direct-Align strategy:
    1. Online rollout: Generate image from noise
    2. Inject noise at intermediate timestep
    3. Inverse/Denoise one step with gradients
    4. Recover image and compute reward (optional, based on reward_model)
    5. Backpropagate through the process
    
    Args:
        args: Training arguments with SRPO parameters
        unet: SDXL UNet (trainable)
        vae: VAE decoder
        text_encoder1: CLIP text encoder
        text_encoder2: OpenCLIP text encoder
        reward_model: Reward model (HPS/PickScore/CLIP), or None to disable reward-based guidance
        scheduler: Noise scheduler
        batch: Training batch data
        global_step: Current training step
        device: torch device
        weight_dtype: Weight dtype
    
    Returns:
        loss: SRPO loss value
    """
    # Get SRPO parameters
    timestep_length = getattr(args, 'srpo_timestep_length', 100)
    discount_pos = getattr(args, 'srpo_discount_pos', [0.1, 0.25])
    discount_inv = getattr(args, 'srpo_discount_inv', [0.3, 0.01])
    train_timestep_range = getattr(args, 'srpo_train_timestep', [5, 25])
    groundtruth_ratio = getattr(args, 'srpo_groundtruth_ratio', 0.9)
    guidance_scale = getattr(args, 'srpo_guidance_scale', 3.5)
    reward_threshold = getattr(args, 'srpo_reward_threshold', 0.7)
    gradient_accumulation_steps = args.gradient_accumulation_steps
    
    # Get custom control words (fall back to defaults if not provided)
    pos_control_words = getattr(args, 'srpo_positive_controls', None)
    neg_control_words = getattr(args, 'srpo_negative_controls', None)
    
    # Discount factors for reward weighting
    discount = torch.linspace(discount_pos[0], discount_pos[1], timestep_length).to(device)
    discount_inversion = torch.linspace(discount_inv[0], discount_inv[1], timestep_length).to(device)
    
    # Get batch data
    latents_shape = batch["latents"].shape
    batch_size = latents_shape[0]
    height, width = latents_shape[2] * 8, latents_shape[3] * 8  # VAE scale factor = 8
    
    # Get text embeddings
    encoder_hidden_states1 = batch["text_encoder_outputs1"]
    encoder_hidden_states2 = batch["text_encoder_outputs2"]
    pooled_prompt_embeds = batch["text_encoder_pool2"]
    
    # Concatenate hidden states for SDXL
    encoder_hidden_states = torch.cat([encoder_hidden_states1, encoder_hidden_states2], dim=-1)
    
    # Get add_time_ids for SDXL
    add_time_ids = batch.get("add_time_ids")
    if add_time_ids is None:
        # Create default add_time_ids if not in batch
        original_size = (height, width)
        target_size = (height, width)
        crops_coords_top_left = (0, 0)
        add_time_ids = torch.tensor([original_size + crops_coords_top_left + target_size]).to(device, dtype=weight_dtype)
        add_time_ids = add_time_ids.repeat(batch_size, 1)
    
    # Expand for CFG if needed
    if guidance_scale > 1.0:
        encoder_hidden_states = torch.cat([encoder_hidden_states] * 2)
        pooled_prompt_embeds = torch.cat([pooled_prompt_embeds] * 2)
        add_time_ids = torch.cat([add_time_ids] * 2)
    
    # Step 1: Online rollout - generate image from noise
    latents = torch.randn(
        (batch_size, 4, height // 8, width // 8),
        device=device,
        dtype=weight_dtype,
    )
    
    with torch.no_grad():
        latents = run_sdxl_sample_step(
            unet=unet,
            scheduler=scheduler,
            latents=latents,
            encoder_hidden_states=encoder_hidden_states,
            pooled_prompt_embeds=pooled_prompt_embeds,
            add_time_ids=add_time_ids,
            guidance_scale=guidance_scale,
            num_inference_steps=50,
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
    
    # Select random timestep for training
    import random
    mid_timestep = random.randint(5, timestep_length - 5)
    k = int((1 - groundtruth_ratio) * timestep_length) + 1
    k = min(min(timestep_length - mid_timestep, k), mid_timestep)
    
    total_loss = 0.0
    
    # Step 2-5: Direct-Align with inversion and denoising branches
    for i in range(gradient_accumulation_steps):
        inversion = i % 2
        
        # Determine timestep range
        if inversion == 0:
            # Denoising branch
            start_t = max(mid_timestep - k, 1)
            end_t = mid_timestep
        else:
            # Inversion branch  
            start_t = mid_timestep
            end_t = min(mid_timestep + k, timestep_length)
        
        # Inject noise at intermediate timestep
        noise = torch.randn_like(latents)
        timestep_tensor = torch.tensor([start_t], device=device)
        noisy_latents = scheduler.add_noise(latents, noise, timestep_tensor)
        
        # Enable gradients for trainable step
        noisy_latents = noisy_latents.detach().requires_grad_(True)
        
        # Forward pass through UNet (with gradients)
        unet.train()
        noise_pred = unet(
            noisy_latents,
            timestep_tensor,
            encoder_hidden_states=encoder_hidden_states[:batch_size],  # Only use positive prompt
            added_cond_kwargs={
                "text_embeds": pooled_prompt_embeds[:batch_size],
                "time_ids": add_time_ids[:batch_size]
            },
            return_dict=False,
        )[0]
        
        # Take one diffusion step (inverse or denoise)
        if inversion == 0:
            # Denoising: move towards clean image
            next_timestep = torch.tensor([end_t], device=device)
            pred_latents = sdxl_euler_step(noise_pred, noisy_latents, timestep_tensor, next_timestep)
        else:
            # Inversion: move towards noisy image
            next_timestep = torch.tensor([end_t], device=device)
            pred_latents = sdxl_euler_step(-noise_pred, noisy_latents, timestep_tensor, next_timestep)
        
        # Check if reward model is enabled
        use_reward_model = getattr(args, 'srpo_use_reward_model', True) and reward_model is not None
        
        if use_reward_model:
            # Decode latents to images for reward computation
            vae.eval()
            with torch.autocast("cuda", dtype=weight_dtype):
                pred_latents_decode = pred_latents / vae.config.scaling_factor
                images = vae.decode(pred_latents_decode, return_dict=False)[0]
                images = (images / 2 + 0.5).clamp(0, 1)
            
            # Compute reward
            with torch.amp.autocast('cuda'):
                if inversion == 1:
                    # Denoising branch: reward for positive style
                    rewards = reward_model.srp_cfg(
                        pos_captions,
                        neg_captions,
                        images,
                        discount[mid_timestep]
                    )
                else:
                    # Inversion branch: penalize negative style
                    rewards = reward_model.srp_cfg(
                        neg_captions,
                        pos_captions,
                        images,
                        discount_inversion[mid_timestep]
                    )
            
            # Compute loss with ReLU threshold (prevents reward hacking)
            loss = F.relu(-rewards + reward_threshold) / gradient_accumulation_steps
            loss = loss.mean()
        else:
            # No reward model: use simple preference loss based on noise prediction quality
            # Compute MSE between predicted noise and actual noise (denoising objective)
            # This encourages better denoising without external reward model bias
            target_latents = latents.detach()  # Use generated latents as target
            
            # Apply preference-based weighting: positive branch gets higher weight
            if inversion == 1:
                # Denoising branch: higher weight (encourage this path)
                weight = 1.0 + discount[mid_timestep]
            else:
                # Inversion branch: lower weight
                weight = 1.0 - discount_inversion[mid_timestep]
            
            # Simple MSE loss weighted by preference
            loss = F.mse_loss(pred_latents, target_latents, reduction='none')
            loss = (loss * weight).mean() / gradient_accumulation_steps
        
        loss.backward()
        total_loss += loss.item()
    
    return total_loss


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
