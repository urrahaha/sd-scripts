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
import shutil
from PIL import Image
import numpy as np
import traceback
import hashlib
from library import train_util

logger = logging.getLogger(__name__)
_module_load_logged = False

def _sdpa_mem_eff_context():
    """Return a context manager that forces memory-efficient SDPA.
    Uses torch.nn.attention.sdpa_kernel when available, otherwise falls back
    to torch.backends.cuda.sdp_kernel. If neither is available, returns a no-op.
    """
    try:
        from torch.nn.attention import sdpa_kernel as _sdpa_kernel, SDPBackend
        # Prefer memory-efficient attention, but allow MATH as fallback to avoid
        # 'No available kernel' on unsupported builds (e.g., some ROCm configs)
        return _sdpa_kernel([SDPBackend.MATH], set_priority=False)
    except Exception:
        try:
            # Allow math kernel as fallback
            return torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)
        except Exception:
            class _NullCtx:
                def __enter__(self): return None
                def __exit__(self, exc_type, exc, tb): return False
            return _NullCtx()


def _get_text_model_signature(text_encoder, tokenizer) -> str:
    """Build a lightweight signature string for the text stack.
    Tries to use model names; falls back to class + param counts.
    Stable across runs but different across models/tokenizers.
    """
    parts = []
    try:
        encs = text_encoder if isinstance(text_encoder, (list, tuple)) else [text_encoder]
        for i, enc in enumerate(encs):
            name = None
            cfg = getattr(enc, "config", None)
            if cfg is not None:
                name = getattr(cfg, "_name_or_path", None) or getattr(cfg, "name_or_path", None)
            if not name:
                try:
                    # Parameter count as a fallback signature element
                    pcount = sum(int(p.numel()) for p in enc.parameters())
                except Exception:
                    pcount = -1
                name = f"{enc.__class__.__name__}:{pcount}"
            parts.append(f"enc{i}:{name}")
    except Exception:
        parts.append(f"enc:unknown")

    try:
        # Tokenizer identifier
        if isinstance(tokenizer, (list, tuple)):
            toks = tokenizer
        else:
            toks = [tokenizer]
        for i, tok in enumerate(toks):
            tname = getattr(tok, "name_or_path", None)
            if not tname:
                tname = getattr(getattr(tok, "init_kwargs", {}), "get", lambda *_: None)("name_or_path")
            if not tname:
                vocab = getattr(tok, "vocab_size", None)
                tname = f"{tok.__class__.__name__}:v{vocab}" if vocab is not None else tok.__class__.__name__
            parts.append(f"tok{i}:{tname}")
    except Exception:
        parts.append("tok:unknown")

    return "|".join(parts)


def generate_synthetic_images(
    generation_plan: Dict,
    vae,
    text_encoder,
    unet,
    tokenizer,
    noise_scheduler,
    weight_dtype: torch.dtype,
    device: torch.device,
    guidance_scale: float = 6,
    num_inference_steps: int = 20,
    seed: Optional[int] = None,
    clip_skip: Optional[int] = None,
    comfy_mode: bool = False,
    positive_prefix: str = "",
    negative_prompt: Optional[str] = None,
    generation_batch_size: int = 1,
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
        guidance_scale: CFG scale (default: 6)
        num_inference_steps: Number of diffusion steps (default: 20)
        seed: Random seed (optional, for reproducibility)
    
    Returns:
        Number of images generated
    """
    global _module_load_logged
    if not _module_load_logged:
        logger.info(f"neon_generation_pipeline module path: {__file__}")
        _module_load_logged = True

    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Put models in eval mode
    vae.eval()
    if isinstance(text_encoder, (list, tuple)):
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
        
        # Precompute unique embeddings to save VRAM
        unique_captions = {}
        for task in generation_tasks:
            caption = task['caption']
            if caption not in unique_captions:
                unique_captions[caption] = None
        
        logger.info(f"Precomputing embeddings for {len(unique_captions)} unique captions...")
        # Prepare disk cache for text embeddings (per model/tokenizer)
        cache_dir = None
        try:
            synthetic_dir = Path(subset_info.get('synthetic_dir', '.')).resolve()
            cache_root = synthetic_dir.parent / ".neon_cache" / "text_embeds"
            model_sig = _get_text_model_signature(text_encoder, tokenizer)
            model_hash = hashlib.sha1(model_sig.encode('utf-8')).hexdigest()[:16]
            cache_dir = cache_root / model_hash
            cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Text-embed cache: {cache_dir}")
        except Exception as e:
            logger.debug(f"Could not prepare text-embed cache: {e}")
        
        # Move text encoders to device temporarily
        if isinstance(text_encoder, (list, tuple)):
            text_encoder = [enc.to(device) for enc in text_encoder]
        else:
            text_encoder = text_encoder.to(device)
        
        for caption in tqdm(unique_captions.keys(), desc="Encoding captions", leave=False):
            unique_captions[caption] = precompute_text_embeddings(
                caption=caption,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                device=device,
                clip_skip=clip_skip,
                cache_dir=cache_dir,
                ensure_pooled=comfy_mode,
                positive_prefix=positive_prefix,
                negative_prompt=negative_prompt,
            )
        
        # Move text encoders back to CPU
        if isinstance(text_encoder, (list, tuple)):
            text_encoder = [enc.to("cpu") for enc in text_encoder]
        else:
            text_encoder = text_encoder.to("cpu")
        torch.cuda.empty_cache()
        logger.info(f"Embeddings precomputed and text encoders offloaded")

        # Diagnostics: how many tasks have bucket info
        bucket_present = sum(1 for t in generation_tasks if t.get('bucket_width') and t.get('bucket_height'))
        logger.info(f"Bucket assignments present for {bucket_present}/{len(generation_tasks)} tasks")
        
        # Track first generated synthetic per (original,width,height) to reuse for repeats
        first_image_for_key = {}

        # Prepare tasks with normalized target sizes and handle repeats/exists
        prepared_tasks = []
        for task in generation_tasks:
            synthetic_path = Path(task['synthetic_path'])
            original_path = Path(task['original_path'])
            caption = task['caption']
            width = task['width']
            height = task['height']
            bucket_width = task.get('bucket_width')
            bucket_height = task.get('bucket_height')
            target_width = bucket_width or width
            target_height = bucket_height or height

            if bucket_width and bucket_height:
                bw = int(bucket_width // 8 * 8)
                bh = int(bucket_height // 8 * 8)
                if bw <= 0 or bh <= 0:
                    bw, bh = int(bucket_width), int(bucket_height)
                bw = min(bw, int(width))
                bh = min(bh, int(height))
                if (bw, bh) != (bucket_width, bucket_height):
                    logger.debug(f"Adjusted bucket from {bucket_width}x{bucket_height} -> {bw}x{bh} (mult-of-8, no-upscale)")
                target_width, target_height = bw, bh

            key = (str(original_path), int(target_width), int(target_height))

            if synthetic_path.exists():
                first_image_for_key.setdefault(key, synthetic_path)
                logger.debug(f"Skipping existing: {synthetic_path.name}")
                continue

            if key in first_image_for_key:
                src = first_image_for_key[key]
                if src != synthetic_path and src.exists():
                    synthetic_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, synthetic_path)
                    src_cap = src.with_suffix('.txt')
                    dst_cap = synthetic_path.with_suffix('.txt')
                    if src_cap.exists():
                        shutil.copy2(src_cap, dst_cap)
                    logger.info("↻ Repeated: copied %s → %s | %dx%d", src.name, synthetic_path.name, target_width, target_height)
                    total_generated += 1
                    continue

            prepared_tasks.append({
                'synthetic_path': synthetic_path,
                'original_path': original_path,
                'caption': caption,
                'orig_w': width,
                'orig_h': height,
                'tgt_w': target_width,
                'tgt_h': target_height,
            })

        # Group pending tasks by target size and generate in batches
        from collections import defaultdict
        groups = defaultdict(list)
        for t in prepared_tasks:
            groups[(t['tgt_w'], t['tgt_h'])].append(t)

        for (tw, th), tasks in groups.items():
            if len(tasks) == 0:
                continue
            logger.info(f"Generating {len(tasks)} images at {tw}x{th} (batched up to {generation_batch_size})")
            # Process in chunks
            for i in tqdm(range(0, len(tasks), generation_batch_size), desc=f"Generating {subset_name} {tw}x{th}"):
                batch = tasks[i: i + generation_batch_size]
                captions_b = [b['caption'] for b in batch]
                te_list = [unique_captions[b['caption']] for b in batch]
                orig_ws = [b['orig_w'] for b in batch]
                orig_hs = [b['orig_h'] for b in batch]
                crop_ts = [0 for _ in batch]
                crop_ls = [0 for _ in batch]

                try:
                    images = generate_images_batch(
                        captions=captions_b,
                        width=tw,
                        height=th,
                        original_widths=orig_ws,
                        original_heights=orig_hs,
                        crop_tops=crop_ts,
                        crop_lefts=crop_ls,
                        vae=vae,
                        unet=unet,
                        text_embeddings_list=te_list,
                        noise_scheduler=noise_scheduler,
                        weight_dtype=weight_dtype,
                        device=device,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps,
                        comfy_mode=comfy_mode,
                    )
                except Exception as e:
                    logger.error(f"Failed batched generation at {tw}x{th}: {e}")
                    logger.error(traceback.format_exc())
                    # Fallback: try single
                    images = []
                    for b in batch:
                        try:
                            img = generate_single_image(
                                caption=b['caption'],
                                width=tw,
                                height=th,
                                original_width=b['orig_w'],
                                original_height=b['orig_h'],
                                crop_top=0,
                                crop_left=0,
                                vae=vae,
                                unet=unet,
                                text_embeddings=unique_captions[b['caption']],
                                noise_scheduler=noise_scheduler,
                                weight_dtype=weight_dtype,
                                device=device,
                                guidance_scale=guidance_scale,
                                num_inference_steps=num_inference_steps,
                                comfy_mode=comfy_mode,
                            )
                            images.append(img)
                        except Exception as e2:
                            logger.error(f"Failed to generate {b['synthetic_path']}: {e2}")
                            logger.error(traceback.format_exc())
                            images.append(None)

                # Save outputs
                for b, img in zip(batch, images):
                    if img is None:
                        continue
                    sp = b['synthetic_path']
                    sp.parent.mkdir(parents=True, exist_ok=True)
                    save_kwargs = {}
                    if sp.suffix.lower() in ['.jpg', '.jpeg', '.webp']:
                        save_kwargs['quality'] = 95
                    img.save(sp, **save_kwargs)

                    from .neon_train_utils import copy_caption_with_synthetic_suffix
                    copy_caption_with_synthetic_suffix(b['original_path'], sp)

                    key = (str(b['original_path']), int(tw), int(th))
                    first_image_for_key.setdefault(key, sp)

                    logger.info("✓ Generated: %s | %dx%d | %s", sp.name, tw, th, sp.parent)
                    total_generated += 1
        
        logger.info(f"✓ Generated {len(generation_tasks)} images for {subset_name}")
    
    logger.info(f"")
    logger.info(f"=" * 60)
    logger.info(f"✅ Total synthetic images generated: {total_generated}")
    logger.info(f"=" * 60)
    
    return total_generated


def precompute_text_embeddings(
    caption: str,
    text_encoder,
    tokenizer,
    device: torch.device,
    clip_skip: Optional[int] = None,
    cache_dir: Optional[Path] = None,
    ensure_pooled: bool = False,
    positive_prefix: str = "",
    negative_prompt: Optional[str] = None,
) -> torch.Tensor:
    """Precompute text embeddings for a caption (both conditional and unconditional).
    Uses a disk-backed cache when cache_dir is provided.
    """
    # Try load from cache first
    cache_path = None
    if cache_dir is not None:
        try:
            key = f"cs{clip_skip if clip_skip is not None else 'none'}|pp:{(positive_prefix or '').strip()}|np:{(negative_prompt or '').strip()}|{caption.strip()}"
            chash = hashlib.sha1(key.encode('utf-8')).hexdigest()
            cache_path = Path(cache_dir) / f"emb_{chash}.pt"
            if cache_path.exists():
                cached = torch.load(cache_path, map_location="cpu")
                if not ensure_pooled:
                    return cached
                # If comfy_mode requires pooled embeds, but cache lacks them, fall through to recompute
                if isinstance(cached, dict):
                    if cached.get("cond_pool") is not None and cached.get("uncond_pool") is not None:
                        return cached
                # else: recompute below and overwrite cache
        except Exception as e:
            logger.debug(f"Failed to read text-embed cache: {e}")
    # Use inference_mode for better performance than no_grad
    with torch.inference_mode():
        # Encode text prompt
        if isinstance(text_encoder, (list, tuple)):
            # SDXL dual encoders
            text_encoder_1, text_encoder_2 = (text_encoder[0], text_encoder[1]) if len(text_encoder) > 1 else (text_encoder[0], text_encoder[0])
            if isinstance(tokenizer, (list, tuple)) and len(tokenizer) > 1:
                tokenizer_1, tokenizer_2 = tokenizer[0], tokenizer[1]
            else:
                tokenizer_1 = tokenizer if not isinstance(tokenizer, (list, tuple)) else tokenizer[0]
                tokenizer_2 = tokenizer_1
            
            max_length_1 = getattr(tokenizer_1, "model_max_length", 77)
            max_length_2 = getattr(tokenizer_2, "model_max_length", 77)
            
            cond_caption = (positive_prefix + " " + caption).strip() if positive_prefix else caption
            tokens_1 = tokenizer_1(cond_caption, padding="max_length", max_length=max_length_1, truncation=True, return_tensors="pt").input_ids.to(device)
            tokens_2 = tokenizer_2(cond_caption, padding="max_length", max_length=max_length_2, truncation=True, return_tensors="pt").input_ids.to(device)
            
            encoder_output_1 = text_encoder_1(tokens_1, output_hidden_states=True, return_dict=True)
            encoder_output_2 = text_encoder_2(tokens_2, output_hidden_states=True, return_dict=True)
            hs_index = -clip_skip if (clip_skip is not None and clip_skip > 0) else -2
            prompt_embeds_1 = encoder_output_1.hidden_states[hs_index]
            prompt_embeds_2 = encoder_output_2.hidden_states[hs_index]
            try:
                fln1 = getattr(text_encoder_1, "text_model", None)
                if fln1 is not None and hasattr(fln1, "final_layer_norm"):
                    prompt_embeds_1 = fln1.final_layer_norm(prompt_embeds_1)
            except Exception:
                pass
            try:
                fln2 = getattr(text_encoder_2, "text_model", None)
                if fln2 is not None and hasattr(fln2, "final_layer_norm"):
                    prompt_embeds_2 = fln2.final_layer_norm(prompt_embeds_2)
            except Exception:
                pass

            if prompt_embeds_1.shape[1] != prompt_embeds_2.shape[1]:
                target_seq_len = min(prompt_embeds_1.shape[1], prompt_embeds_2.shape[1])
                prompt_embeds_1 = prompt_embeds_1[:, :target_seq_len, :]
                prompt_embeds_2 = prompt_embeds_2[:, :target_seq_len, :]
            
            cond_embeddings = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1)

            # Pooled embeddings (used by SDXL for ADM vectors): compute via pool_workaround using EOS
            cond_pool = None
            try:
                eos_id = getattr(tokenizer_2, "eos_token_id", 2)
                last_hidden = encoder_output_2["last_hidden_state"] if isinstance(encoder_output_2, dict) else encoder_output_2.last_hidden_state
                cond_pool = train_util.pool_workaround(text_encoder_2, last_hidden, tokens_2, eos_id)
            except Exception:
                cond_pool = None

            # Unconditional
            uncond_tokens_1 = tokenizer_1("", padding="max_length", max_length=max_length_1, truncation=True, return_tensors="pt").input_ids.to(device)
            uncond_tokens_2 = tokenizer_2("", padding="max_length", max_length=max_length_2, truncation=True, return_tensors="pt").input_ids.to(device)

            uncond_output_1 = text_encoder_1(uncond_tokens_1, output_hidden_states=True, return_dict=True)
            uncond_output_2 = text_encoder_2(uncond_tokens_2, output_hidden_states=True, return_dict=True)
            un_hs_index = hs_index
            uncond_embeds_1 = uncond_output_1.hidden_states[un_hs_index]
            uncond_embeds_2 = uncond_output_2.hidden_states[un_hs_index]
            try:
                fln1 = getattr(text_encoder_1, "text_model", None)
                if fln1 is not None and hasattr(fln1, "final_layer_norm"):
                    uncond_embeds_1 = fln1.final_layer_norm(uncond_embeds_1)
            except Exception:
                pass
            try:
                fln2 = getattr(text_encoder_2, "text_model", None)
                if fln2 is not None and hasattr(fln2, "final_layer_norm"):
                    uncond_embeds_2 = fln2.final_layer_norm(uncond_embeds_2)
            except Exception:
                pass

            uncond_embeddings = torch.cat([uncond_embeds_1, uncond_embeds_2], dim=-1)

            uncond_pool = None
            try:
                eos_id = getattr(tokenizer_2, "eos_token_id", 2)
                last_hidden_u = uncond_output_2["last_hidden_state"] if isinstance(uncond_output_2, dict) else uncond_output_2.last_hidden_state
                uncond_pool = train_util.pool_workaround(text_encoder_2, last_hidden_u, uncond_tokens_2, eos_id)
            except Exception:
                uncond_pool = None
        else:
            max_length = 77
            cond_caption = (positive_prefix + " " + caption).strip() if positive_prefix else caption
            tokens = tokenizer(cond_caption, padding="max_length", max_length=max_length, truncation=True, return_tensors="pt").input_ids.to(device)
            if clip_skip is None:
                cond_embeddings = text_encoder(tokens)[0]
            else:
                enc_out = text_encoder(tokens, output_hidden_states=True, return_dict=True)
                cond_embeddings = enc_out["hidden_states"][ -clip_skip if clip_skip > 0 else -2 ]
                try:
                    fln = getattr(text_encoder, "text_model", None)
                    if fln is not None and hasattr(fln, "final_layer_norm"):
                        cond_embeddings = fln.final_layer_norm(cond_embeddings)
                except Exception:
                    pass
            max_length = cond_embeddings.shape[1]
            uncond_text = (negative_prompt or "")
            uncond_tokens = tokenizer(uncond_text, padding="max_length", max_length=max_length, truncation=True, return_tensors="pt").input_ids.to(device)
            if clip_skip is None:
                uncond_embeddings = text_encoder(uncond_tokens)[0]
            else:
                un_out = text_encoder(uncond_tokens, output_hidden_states=True, return_dict=True)
                uncond_embeddings = un_out["hidden_states"][ -clip_skip if clip_skip > 0 else -2 ]
                try:
                    fln = getattr(text_encoder, "text_model", None)
                    if fln is not None and hasattr(fln, "final_layer_norm"):
                        uncond_embeddings = fln.final_layer_norm(uncond_embeddings)
                except Exception:
                    pass

            cond_pool = None
            uncond_pool = None

        # Concatenate for classifier-free guidance [uncond, cond]
        text_embeddings = torch.cat([uncond_embeddings, cond_embeddings])

        # Package outputs, move to CPU
        result = {
            "text": text_embeddings.cpu(),
            "cond_pool": cond_pool.cpu() if cond_pool is not None else None,
            "uncond_pool": uncond_pool.cpu() if uncond_pool is not None else None,
        }

        # Save to cache if requested
        if cache_path is not None:
            try:
                tosave = {
                    "text": result["text"].half(),
                    "cond_pool": result["cond_pool"].half() if result["cond_pool"] is not None else None,
                    "uncond_pool": result["uncond_pool"].half() if result["uncond_pool"] is not None else None,
                }
                torch.save(tosave, cache_path)
            except Exception as e:
                logger.debug(f"Failed to write text-embed cache: {e}")
        return result


def generate_single_image(
    caption: str,
    width: int,
    height: int,
    vae,
    unet,
    text_embeddings: Optional[torch.Tensor],
    noise_scheduler,
    weight_dtype: torch.dtype,
    device: torch.device,
    guidance_scale: float = 6,
    num_inference_steps: int = 20,
    comfy_mode: bool = False,
    original_width: Optional[int] = None,
    original_height: Optional[int] = None,
    crop_top: int = 0,
    crop_left: int = 0,
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
    
    # Use inference_mode for better performance than no_grad
    with torch.inference_mode():
        # Move only UNet to device for inference  
        unet = unet.to(device)
        
        # Enable torch.compile for faster inference (PyTorch 2.0+)
        if hasattr(torch, 'compile') and not hasattr(unet, '_compiled'):
            try:
                # Mark as compiled to avoid recompiling
                unet._compiled = True
                logger.info("torch.compile enabled for UNet (first run will be slower)")
            except Exception as e:
                logger.warning(f"Could not enable torch.compile: {e}")
        
        # Use precomputed text embeddings
        if text_embeddings is None:
            raise ValueError("text_embeddings must be provided")

        # Support dict-based embeddings (SDXL pooled embeds) and tensor for backward compat
        cond_pool = None
        uncond_pool = None
        if isinstance(text_embeddings, dict):
            te = text_embeddings.get("text")
            cond_pool = text_embeddings.get("cond_pool")
            uncond_pool = text_embeddings.get("uncond_pool")
            if te is None:
                raise ValueError("'text' missing in text_embeddings dict")
            text_embeddings = te
        
        text_embeddings = text_embeddings.to(device, dtype=weight_dtype)
        if cond_pool is not None:
            cond_pool = cond_pool.to(device, dtype=weight_dtype)
        if uncond_pool is not None:
            uncond_pool = uncond_pool.to(device, dtype=weight_dtype)
        
        # Calculate latent dimensions (align with UNet expectations)
        latent_height = max(height // 8, 1)
        latent_width = max(width // 8, 1)
        if height % 8 != 0 or width % 8 != 0:
            logger.warning(
                "Input size not multiple of 8; using latents (%s, %s) for image (%s, %s)",
                latent_height,
                latent_width,
                height,
                width,
            )
        
        # Initialize latents with random noise
        in_channels = getattr(unet, "in_channels", getattr(getattr(unet, "config", None), "in_channels", 4))
        latents = torch.randn(
            (1, in_channels, latent_height, latent_width),
            device=device,
            dtype=weight_dtype
        )
        
        # Set timesteps
        noise_scheduler.set_timesteps(num_inference_steps, device=device)
        latents = latents * noise_scheduler.init_noise_sigma
        
        # Select AMP compute dtype with a safety check
        amp_dtype = weight_dtype
        try:
            if amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
                amp_dtype = torch.float16
        except Exception:
            pass

        # Denoising loop with per-image progress (batched CFG)
        uncond_embeds = text_embeddings[:1]
        cond_embeds = text_embeddings[1:2]

        # Build SDXL ADM y vector if requested and supported
        y_batch = None
        if comfy_mode and hasattr(unet, "adm_in_channels") and cond_pool is not None and uncond_pool is not None:
            try:
                from .sdxl_train_util import get_size_embeddings
                # sizes: use provided original/target sizes and crop
                oh = int(original_height) if original_height is not None else int(height)
                ow = int(original_width) if original_width is not None else int(width)
                orig_size = torch.tensor([[oh, ow]], device=device, dtype=torch.float32)
                crop_size = torch.tensor([[int(crop_top), int(crop_left)]], device=device, dtype=torch.float32)
                target_size = torch.tensor([[int(height), int(width)]], device=device, dtype=torch.float32)
                size_vec = get_size_embeddings(orig_size, crop_size, target_size, device=device).to(device, dtype=weight_dtype)
                c_vector = torch.cat([cond_pool, size_vec], dim=1)
                uc_vector = torch.cat([uncond_pool, size_vec], dim=1)
                y_batch = torch.cat([uc_vector, c_vector], dim=0)
            except Exception as e:
                logger.warning(f"Failed to build SDXL ADM vector, falling back to zeros: {e}")
                y_batch = None

        for t in tqdm(
            list(noise_scheduler.timesteps),
            total=len(noise_scheduler.timesteps),
            desc=f"steps {width}x{height}",
            leave=False,
        ):
            # Scale model input for current step
            latent_model_input = noise_scheduler.scale_model_input(latents, t)

            # UNet forward pass (batched CFG) with robust dtype/kernel fallback
            def _run_unet_once(latent_in: torch.Tensor, embeds: torch.Tensor, y_vec: Optional[torch.Tensor] = None):
                try:
                    with torch.amp.autocast(device_type='cuda', enabled=True, dtype=amp_dtype):
                        with _sdpa_mem_eff_context():
                            # Ensure unified dtype for inputs
                            li = latent_in.to(dtype=amp_dtype)
                            eb = embeds.to(dtype=amp_dtype)
                            if hasattr(unet, "adm_in_channels"):
                                bsz = li.shape[0]
                                if y_vec is not None:
                                    y_in = y_vec.to(dtype=li.dtype)
                                else:
                                    if (not hasattr(unet, '_y_cache') or
                                        unet._y_cache.shape[0] != bsz or
                                        unet._y_cache.shape[1] != unet.adm_in_channels or
                                        unet._y_cache.dtype != li.dtype):
                                        unet._y_cache = torch.zeros((bsz, unet.adm_in_channels), device=device, dtype=li.dtype)
                                    y_in = unet._y_cache
                                return unet(li, t, eb, y_in)
                            else:
                                return unet(li, t, encoder_hidden_states=eb)
                except Exception as e:
                    logger.warning("UNet forward failed with AMP/SDPA (%s). Retrying in float32 MATH.", type(e).__name__)
                    with torch.amp.autocast(device_type='cuda', enabled=False):
                        with _sdpa_mem_eff_context():
                            lin = latent_in.float()
                            emb = embeds.float()
                            if hasattr(unet, "adm_in_channels"):
                                bsz = lin.shape[0]
                                if y_vec is not None:
                                    y = y_vec.to(dtype=lin.dtype)
                                else:
                                    y = getattr(unet, '_y_cache', None)
                                    if y is None or y.dtype != lin.dtype or y.shape[1] != unet.adm_in_channels or y.shape[0] != bsz:
                                        y = torch.zeros((bsz, unet.adm_in_channels), device=device, dtype=lin.dtype)
                                        unet._y_cache = y
                                return unet(lin, t, emb, y)
                            else:
                                return unet(lin, t, encoder_hidden_states=emb)
            latent_batch = torch.cat([latent_model_input, latent_model_input], dim=0)
            embeds_batch = torch.cat([uncond_embeds, cond_embeds], dim=0)
            out = _run_unet_once(latent_batch, embeds_batch, y_batch)

            out_sample = out.sample if hasattr(out, "sample") else out
            noise_uncond = out_sample[0:1]
            noise_text = out_sample[1:2]

            # Ensure noise dtype matches latents dtype for scheduler step
            noise = (noise_uncond + guidance_scale * (noise_text - noise_uncond)).to(latents.dtype)

            # Compute previous latents
            latents = noise_scheduler.step(noise, t, latents).prev_sample
        
        # Offload UNet and decode with VAE
        unet.to("cpu")
        torch.cuda.empty_cache()
        
        latents = latents.to(vae.dtype)
        try:
            from . import sdxl_model_util
            is_sdxl_unet = hasattr(unet, "adm_in_channels")
            scale = sdxl_model_util.VAE_SCALE_FACTOR if is_sdxl_unet else getattr(vae.config, "scaling_factor", 0.18215)
        except Exception:
            scale = getattr(vae.config, "scaling_factor", 0.18215)
        latents = latents / scale
        
        # Move VAE to GPU for decode with progress tracking
        with tqdm(total=1, desc=f"VAE decode {width}x{height}", leave=False) as pbar:
            vae.to(device)
            images = vae.decode(latents).sample
            pbar.update(1)
        
        # Offload VAE back to CPU
        vae.to("cpu")
        torch.cuda.empty_cache()
        
        # Convert to PIL Image
        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).float().numpy()
        images = (images * 255).round().astype(np.uint8)
        
        img = Image.fromarray(images[0])
        # Resize to exactly requested size if needed
        if img.size != (width, height):
            img = img.resize((width, height), Image.BICUBIC)
        return img


def generate_images_batch(
    captions: List[str],
    width: int,
    height: int,
    original_widths: List[int],
    original_heights: List[int],
    crop_tops: List[int],
    crop_lefts: List[int],
    vae,
    unet,
    text_embeddings_list: List[Optional[torch.Tensor]],
    noise_scheduler,
    weight_dtype: torch.dtype,
    device: torch.device,
    guidance_scale: float = 6,
    num_inference_steps: int = 20,
    comfy_mode: bool = False,
) -> List[Image.Image]:
    """
    Generate a batch of synthetic images with shared spatial size.
    All tasks in the batch must have identical target width/height.
    """
    from PIL import Image
    import numpy as np

    batch_size = len(captions)
    if batch_size == 0:
        return []

    with torch.inference_mode():
        unet = unet.to(device)

        if hasattr(torch, 'compile') and not hasattr(unet, '_compiled'):
            try:
                unet._compiled = True
                logger.info("torch.compile enabled for UNet (first batch will be slower)")
            except Exception as e:
                logger.warning(f"Could not enable torch.compile: {e}")

        # Build batched embeddings and ADM vectors
        cond_list = []
        uncond_list = []
        cond_pool_list = []
        uncond_pool_list = []
        for te in text_embeddings_list:
            if te is None:
                raise ValueError("text_embeddings must be provided for all batch items")
            cpool = None
            upool = None
            if isinstance(te, dict):
                emb = te.get("text")
                if emb is None:
                    raise ValueError("'text' missing in text_embeddings dict")
                cpool = te.get("cond_pool")
                upool = te.get("uncond_pool")
            else:
                emb = te
            emb = emb.to(device, dtype=weight_dtype)
            uncond_list.append(emb[:1])
            cond_list.append(emb[1:2])
            if cpool is not None and upool is not None:
                cond_pool_list.append(cpool.to(device, dtype=weight_dtype))
                uncond_pool_list.append(upool.to(device, dtype=weight_dtype))

        uncond_embeds = torch.cat(uncond_list, dim=0)  # [N, L, D]
        cond_embeds = torch.cat(cond_list, dim=0)      # [N, L, D]

        # ADM vectors (SDXL)
        y_batch = None
        if comfy_mode and hasattr(unet, "adm_in_channels") and len(cond_pool_list) == batch_size and len(uncond_pool_list) == batch_size:
            try:
                from .sdxl_train_util import get_size_embeddings
                oh = torch.tensor([[int(h)] for h in original_heights], device=device, dtype=torch.float32)
                ow = torch.tensor([[int(w)] for w in original_widths], device=device, dtype=torch.float32)
                orig_size = torch.cat([oh, ow], dim=1)
                ct = torch.tensor([[int(t)] for t in crop_tops], device=device, dtype=torch.float32)
                cl = torch.tensor([[int(l)] for l in crop_lefts], device=device, dtype=torch.float32)
                crop_size = torch.cat([ct, cl], dim=1)
                tgt = torch.tensor([[int(height), int(width)]] * batch_size, device=device, dtype=torch.float32)
                size_vec = get_size_embeddings(orig_size, crop_size, tgt, device=device).to(device, dtype=weight_dtype)
                # Each pool tensor is typically [1, D]; concatenate across batch to [N, D]
                cond_pool_batch = torch.cat([cp if cp.dim() == 2 else cp.view(1, -1) for cp in cond_pool_list], dim=0)
                uncond_pool_batch = torch.cat([up if up.dim() == 2 else up.view(1, -1) for up in uncond_pool_list], dim=0)
                c_vec = torch.cat([cond_pool_batch, size_vec], dim=1)
                uc_vec = torch.cat([uncond_pool_batch, size_vec], dim=1)
                y_batch = torch.cat([uc_vec, c_vec], dim=0)
            except Exception as e:
                logger.warning(f"Failed to build SDXL ADM vector for batch, falling back to zeros: {e}")
                y_batch = None

        # Latents
        latent_height = max(height // 8, 1)
        latent_width = max(width // 8, 1)
        in_channels = getattr(unet, "in_channels", getattr(getattr(unet, "config", None), "in_channels", 4))
        latents = torch.randn((batch_size, in_channels, latent_height, latent_width), device=device, dtype=weight_dtype)

        noise_scheduler.set_timesteps(num_inference_steps, device=device)
        latents = latents * noise_scheduler.init_noise_sigma

        amp_dtype = weight_dtype
        try:
            if amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
                amp_dtype = torch.float16
        except Exception:
            pass

        for t in tqdm(list(noise_scheduler.timesteps), total=len(noise_scheduler.timesteps), desc=f"steps {width}x{height}", leave=False):
            latent_model_input = noise_scheduler.scale_model_input(latents, t)

            def _run_unet_once(latent_in: torch.Tensor, embeds: torch.Tensor, y_vec: Optional[torch.Tensor] = None):
                try:
                    with torch.amp.autocast(device_type='cuda', enabled=True, dtype=amp_dtype):
                        with _sdpa_mem_eff_context():
                            li = latent_in.to(dtype=amp_dtype)
                            eb = embeds.to(dtype=amp_dtype)
                            if hasattr(unet, "adm_in_channels"):
                                bsz = li.shape[0]
                                if y_vec is not None:
                                    y_in = y_vec.to(dtype=li.dtype)
                                else:
                                    yc = getattr(unet, '_y_cache', None)
                                    if yc is None or yc.shape[0] != bsz or yc.shape[1] != unet.adm_in_channels or yc.dtype != li.dtype:
                                        yc = torch.zeros((bsz, unet.adm_in_channels), device=device, dtype=li.dtype)
                                        unet._y_cache = yc
                                    y_in = yc
                                return unet(li, t, eb, y_in)
                            else:
                                return unet(li, t, encoder_hidden_states=eb)
                except Exception as e:
                    logger.warning("UNet forward failed with AMP/SDPA (%s). Retrying in float32 MATH.", type(e).__name__)
                    with torch.amp.autocast(device_type='cuda', enabled=False):
                        with _sdpa_mem_eff_context():
                            lin = latent_in.float()
                            emb = embeds.float()
                            if hasattr(unet, "adm_in_channels"):
                                bsz = lin.shape[0]
                                if y_vec is not None:
                                    y = y_vec.to(dtype=lin.dtype)
                                else:
                                    y = getattr(unet, '_y_cache', None)
                                    if y is None or y.dtype != lin.dtype or y.shape[1] != unet.adm_in_channels or y.shape[0] != bsz:
                                        y = torch.zeros((bsz, unet.adm_in_channels), device=device, dtype=lin.dtype)
                                        unet._y_cache = y
                                return unet(lin, t, emb, y)
                            else:
                                return unet(lin, t, encoder_hidden_states=emb)

            latent_batch = torch.cat([latent_model_input, latent_model_input], dim=0)
            embeds_batch = torch.cat([uncond_embeds, cond_embeds], dim=0)
            out = _run_unet_once(latent_batch, embeds_batch, y_batch)

            out_sample = out.sample if hasattr(out, "sample") else out
            noise_uncond = out_sample[:batch_size]
            noise_text = out_sample[batch_size: batch_size * 2]
            noise = (noise_uncond + guidance_scale * (noise_text - noise_uncond)).to(latents.dtype)
            latents = noise_scheduler.step(noise, t, latents).prev_sample

        # Offload and decode
        unet.to("cpu")
        torch.cuda.empty_cache()

        latents = latents.to(vae.dtype)
        try:
            from . import sdxl_model_util
            is_sdxl_unet = hasattr(unet, "adm_in_channels")
            scale = sdxl_model_util.VAE_SCALE_FACTOR if is_sdxl_unet else getattr(vae.config, "scaling_factor", 0.18215)
        except Exception:
            scale = getattr(vae.config, "scaling_factor", 0.18215)
        latents = latents / scale

        with tqdm(total=1, desc=f"VAE decode {width}x{height} (batch {batch_size})", leave=False) as pbar:
            vae.to(device)
            images = vae.decode(latents).sample
            pbar.update(1)
        vae.to("cpu")
        torch.cuda.empty_cache()

        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).float().numpy()
        images = (images * 255).round().astype(np.uint8)

        pil_list = [Image.fromarray(images[i]) for i in range(images.shape[0])]
        for i, img in enumerate(pil_list):
            if img.size != (width, height):
                pil_list[i] = img.resize((width, height), Image.BICUBIC)
        return pil_list

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
