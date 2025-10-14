"""
Neon Synthetic Image Generation Pipeline

Generates synthetic images using trained LoRA + base model.
Mimics original image dimensions and uses original captions.
"""

import torch
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


def generate_synthetic_images(
    generation_plan: Dict,
    vae,
    text_encoder,
    unet,
    tokenizer,
    noise_scheduler,
    weight_dtype: torch.dtype,
    device: torch.device,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 28,
    seed: Optional[int] = None,
) -> int:
    """
    Generate synthetic images based on generation plan.
    
    For each task in the plan:
    - Reads original image dimensions and caption
    - Generates synthetic image matching those dimensions
    - Saves as <original_name>_synthetic.<ext>
    - Copies caption as <original_name>_synthetic.txt
    
    Args:
        generation_plan: Dict from replicate_dataset_structure()
        vae: VAE model
        text_encoder: Text encoder (or tuple for SDXL)
        unet: UNet model
        tokenizer: Tokenizer (or tuple for SDXL)
        noise_scheduler: Noise scheduler for diffusion
        weight_dtype: Weight dtype (torch.float16, torch.bfloat16, etc.)
        device: Device to run on
        guidance_scale: CFG scale (default: 7.5)
        num_inference_steps: Number of diffusion steps (default: 28)
        seed: Random seed (optional, for reproducibility)
    
    Returns:
        Number of images generated
    """
    from PIL import Image
    import numpy as np
    
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Put models in eval mode
    vae.eval()
    if isinstance(text_encoder, tuple):
        for encoder in text_encoder:
            encoder.eval()
    else:
        text_encoder.eval()
    unet.eval()
    
    total_generated = 0
    
    # Process each subset
    for subset_name, subset_info in generation_plan.items():
        generation_tasks = subset_info.get('generation_tasks', [])
        
        if not generation_tasks:
            continue
        
        logger.info(f"")
        logger.info(f"Generating synthetic images for: {subset_name}")
        logger.info(f"Tasks: {len(generation_tasks)}")
        
        for task in tqdm(generation_tasks, desc=f"Generating {subset_name}"):
            try:
                synthetic_path = Path(task['synthetic_path'])
                original_path = Path(task['original_path'])
                caption = task['caption']
                width = task['width']
                height = task['height']
                
                # Skip if already exists
                if synthetic_path.exists():
                    logger.debug(f"Skipping existing: {synthetic_path.name}")
                    continue
                
                # Generate synthetic image
                image = generate_single_image(
                    caption=caption,
                    width=width,
                    height=height,
                    vae=vae,
                    text_encoder=text_encoder,
                    unet=unet,
                    tokenizer=tokenizer,
                    noise_scheduler=noise_scheduler,
                    weight_dtype=weight_dtype,
                    device=device,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                )
                
                # Save synthetic image
                synthetic_path.parent.mkdir(parents=True, exist_ok=True)
                image.save(synthetic_path, quality=95 if synthetic_path.suffix.lower() in ['.jpg', '.jpeg'] else None)
                
                # Copy caption with _synthetic suffix
                from .neon_train_utils import copy_caption_with_synthetic_suffix
                copy_caption_with_synthetic_suffix(original_path, synthetic_path)
                
                total_generated += 1
                
            except Exception as e:
                logger.error(f"Failed to generate {task['synthetic_path']}: {e}")
                continue
        
        logger.info(f"✓ Generated {len(generation_tasks)} images for {subset_name}")
    
    logger.info(f"")
    logger.info(f"=" * 60)
    logger.info(f"✅ Total synthetic images generated: {total_generated}")
    logger.info(f"=" * 60)
    
    return total_generated


def generate_single_image(
    caption: str,
    width: int,
    height: int,
    vae,
    text_encoder,
    unet,
    tokenizer,
    noise_scheduler,
    weight_dtype: torch.dtype,
    device: torch.device,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 28,
) -> Image.Image:
    """
    Generate a single synthetic image using diffusion pipeline.
    
    Args:
        caption: Text caption/prompt
        width: Target image width
        height: Target image height
        (other args same as generate_synthetic_images)
    
    Returns:
        PIL Image
    """
    from PIL import Image
    import numpy as np
    
    with torch.no_grad():
        # Encode text prompt
        if isinstance(text_encoder, tuple):
            # SDXL has multiple text encoders
            text_encoder_1, text_encoder_2 = text_encoder
            tokenizer_1, tokenizer_2 = tokenizer
            
            # Tokenize for both encoders
            tokens_1 = tokenizer_1(
                caption,
                padding="max_length",
                max_length=tokenizer_1.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(device)
            
            tokens_2 = tokenizer_2(
                caption,
                padding="max_length",
                max_length=tokenizer_2.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(device)
            
            # Encode with both encoders
            encoder_output_1 = text_encoder_1(tokens_1, output_hidden_states=True)
            encoder_output_2 = text_encoder_2(tokens_2, output_hidden_states=True)
            
            # Concatenate embeddings (SDXL specific)
            text_embeddings = torch.cat([
                encoder_output_1.hidden_states[-2],
                encoder_output_2.hidden_states[-2]
            ], dim=-1)
            
        else:
            # SD 1.5 / SD 2.x
            tokens = tokenizer(
                caption,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(device)
            
            text_embeddings = text_encoder(tokens)[0]
        
        # Encode unconditional (for CFG)
        if isinstance(tokenizer, tuple):
            tokenizer_1, tokenizer_2 = tokenizer
            uncond_tokens_1 = tokenizer_1(
                "",
                padding="max_length",
                max_length=tokenizer_1.model_max_length,
                return_tensors="pt",
            ).input_ids.to(device)
            
            uncond_tokens_2 = tokenizer_2(
                "",
                padding="max_length",
                max_length=tokenizer_2.model_max_length,
                return_tensors="pt",
            ).input_ids.to(device)
            
            text_encoder_1, text_encoder_2 = text_encoder
            uncond_output_1 = text_encoder_1(uncond_tokens_1, output_hidden_states=True)
            uncond_output_2 = text_encoder_2(uncond_tokens_2, output_hidden_states=True)
            
            uncond_embeddings = torch.cat([
                uncond_output_1.hidden_states[-2],
                uncond_output_2.hidden_states[-2]
            ], dim=-1)
        else:
            uncond_tokens = tokenizer(
                "",
                padding="max_length",
                max_length=tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids.to(device)
            
            uncond_embeddings = text_encoder(uncond_tokens)[0]
        
        # Concatenate for classifier-free guidance
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
        # Calculate latent dimensions
        latent_height = height // 8
        latent_width = width // 8
        
        # Initialize latents with random noise
        latents = torch.randn(
            (1, unet.config.in_channels, latent_height, latent_width),
            device=device,
            dtype=weight_dtype
        )
        
        # Set timesteps
        noise_scheduler.set_timesteps(num_inference_steps, device=device)
        latents = latents * noise_scheduler.init_noise_sigma
        
        # Denoising loop
        for t in noise_scheduler.timesteps:
            # Expand latents for CFG
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)
            
            # Predict noise
            noise_pred = unet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings,
            ).sample
            
            # Perform classifier-free guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Compute previous latents
            latents = noise_scheduler.step(noise_pred, t, latents).prev_sample
        
        # Decode latents to image
        latents = latents.to(vae.dtype)
        latents = 1 / vae.config.scaling_factor * latents
        images = vae.decode(latents).sample
        
        # Convert to PIL Image
        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).float().numpy()
        images = (images * 255).round().astype(np.uint8)
        
        return Image.fromarray(images[0])


def test_generation_pipeline(
    generation_plan: Dict,
    num_test_images: int = 3
):
    """
    Test the generation pipeline with placeholder generation.
    Useful for testing the pipeline without actual model loading.
    
    Args:
        generation_plan: Generation plan from replicate_dataset_structure()
        num_test_images: Number of test images to generate per subset
    """
    from PIL import Image, ImageDraw, ImageFont
    import random
    
    logger.info("=" * 60)
    logger.info("TEST MODE: Generating placeholder synthetic images")
    logger.info("=" * 60)
    
    total_generated = 0
    
    for subset_name, subset_info in generation_plan.items():
        generation_tasks = subset_info.get('generation_tasks', [])[:num_test_images]
        
        if not generation_tasks:
            continue
        
        logger.info(f"\nGenerating test images for: {subset_name}")
        
        for task in generation_tasks:
            synthetic_path = Path(task['synthetic_path'])
            original_path = Path(task['original_path'])
            caption = task['caption']
            width = task['width']
            height = task['height']
            
            # Create placeholder image with colored background
            color = (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200))
            img = Image.new('RGB', (width, height), color=color)
            
            # Draw text
            draw = ImageDraw.Draw(img)
            text = f"SYNTHETIC\n{width}x{height}\n{caption[:50]}"
            draw.text((10, 10), text, fill=(255, 255, 255))
            
            # Save
            synthetic_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(synthetic_path, quality=95 if synthetic_path.suffix.lower() in ['.jpg', '.jpeg'] else None)
            
            # Copy caption
            from .neon_train_utils import copy_caption_with_synthetic_suffix
            copy_caption_with_synthetic_suffix(original_path, synthetic_path)
            
            total_generated += 1
            logger.info(f"  Generated: {synthetic_path.name}")
    
    logger.info(f"\n✓ Generated {total_generated} test images")
