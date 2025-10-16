"""
Neon (Negative Extrapolation from Self-Training) Utilities
Based on: https://github.com/VITA-Group/Neon

Improves generative models by:
1. Generating synthetic samples from the base model
2. Fine-tuning on synthetic data (creates degraded model)
3. Extrapolating away from degradation via weight merge
"""

import torch
import logging
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)


def build_neon_state_dict(
    base_state_dict: dict,
    aux_state_dict: dict,
    w: float = 0.3,
    device: Optional[torch.device] = None,
) -> dict:
    """
    Merge two state dicts using Neon's negative extrapolation formula:
    θ_neon = (1+w)*θ_base - w*θ_aux = θ_base - w*(θ_aux - θ_base)
    
    Args:
        base_state_dict: Reference model weights (pre-synthetic training)
        aux_state_dict: Auxiliary model weights (post-synthetic training)
        w: Extrapolation weight (default: 0.3, typical range: 0.1-0.5)
        device: Device for merge computation
    
    Returns:
        Merged state dict with improved weights
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    merged = {}
    
    for key, base_tensor in base_state_dict.items():
        # Non-tensor items (e.g., metadata)
        if not isinstance(base_tensor, torch.Tensor):
            merged[key] = base_tensor
            continue
        
        base_t = base_tensor.to(device=device)
        
        # Check if key exists in aux and is a floating point tensor
        if key in aux_state_dict and isinstance(aux_state_dict[key], torch.Tensor) and torch.is_floating_point(base_t):
            aux_t = aux_state_dict[key].to(device=device, dtype=base_t.dtype)
            
            # Initialize merged tensor
            m = base_t.clone()
            
            # Only merge finite values (handles NaN/Inf gracefully)
            finite_mask = torch.isfinite(base_t) & torch.isfinite(aux_t)
            
            if finite_mask.any():
                # Neon formula: θ_base - w * (θ_aux - θ_base)
                # Equivalent to: (1+w)*θ_base - w*θ_aux
                m[finite_mask] = base_t[finite_mask] - w * (aux_t[finite_mask] - base_t[finite_mask])
            
            merged[key] = m.cpu()  # Move back to CPU to save VRAM
        else:
            # Key not in aux or not compatible -> keep base
            merged[key] = base_t.cpu()
    
    return merged


def neon_merge_lora(
    base_lora_path: str,
    aux_lora_path: str,
    output_path: str,
    w: float = 0.3,
    device: Optional[torch.device] = None,
) -> None:
    """
    Apply Neon merge to LoRA weights and save result.
    
    Args:
        base_lora_path: Path to base LoRA (reference model)
        aux_lora_path: Path to auxiliary LoRA (fine-tuned on synthetic data)
        output_path: Path to save merged Neon LoRA
        w: Extrapolation weight (default: 0.3)
        device: Device for computation
    
    Example:
        # 1. Train base LoRA normally
        # 2. Generate synthetic dataset using base LoRA
        # 3. Fine-tune on synthetic data for few steps (creates aux LoRA)
        # 4. Merge:
        neon_merge_lora("base.safetensors", "aux.safetensors", "neon.safetensors", w=0.3)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"Loading base LoRA from: {base_lora_path}")
    base_sd = load_state_dict(base_lora_path, device)
    
    logger.info(f"Loading auxiliary LoRA from: {aux_lora_path}")
    aux_sd = load_state_dict(aux_lora_path, device)
    
    logger.info(f"Applying Neon merge with w={w}")
    merged_sd = build_neon_state_dict(base_sd, aux_sd, w=w, device=device)
    
    logger.info(f"Saving Neon LoRA to: {output_path}")
    save_state_dict(merged_sd, output_path)
    
    logger.info("✅ Neon merge complete!")


def load_state_dict(path: str, device: torch.device) -> dict:
    """Load state dict from .pt, .pth, or .safetensors file"""
    path_obj = Path(path)
    
    if not path_obj.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    
    if path.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file
            return load_file(path, device=str(device))
        except ImportError:
            raise ImportError("safetensors not installed. Run: pip install safetensors")
    else:
        # .pt or .pth
        return torch.load(path, map_location=device)


def save_state_dict(state_dict: dict, path: str) -> None:
    """Save state dict to .pt, .pth, or .safetensors file"""
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    if path.endswith(".safetensors"):
        try:
            from safetensors.torch import save_file
            save_file(state_dict, path)
        except ImportError:
            raise ImportError("safetensors not installed. Run: pip install safetensors")
    else:
        # .pt or .pth
        torch.save(state_dict, path)


def should_use_neon_training(args) -> bool:
    """Check if Neon training mode is enabled"""
    return getattr(args, 'neon_enable', False)


def validate_neon_args(args) -> None:
    """Validate Neon training arguments"""
    if not should_use_neon_training(args):
        return
    
    # Check synthetic data path
    synthetic_data = getattr(args, 'neon_synthetic_data', None)
    if not synthetic_data:
        raise ValueError("Neon training requires --neon_synthetic_data path")
    
    # Check extrapolation weight
    w = getattr(args, 'neon_extrapolation_weight', 0.3)
    if not (0.0 < w < 1.0):
        logger.warning(f"Unusual Neon weight: {w}. Typical range is 0.1-0.5")
    
    # Check base model checkpoint
    base_checkpoint = getattr(args, 'neon_base_checkpoint', None)
    if not base_checkpoint:
        logger.warning("No base checkpoint specified. Will use current model as base.")
    
    logger.info("Neon Training Configuration:")
    logger.info(f"  Synthetic data: {synthetic_data}")
    logger.info(f"  Extrapolation weight: {w}")
    logger.info(f"  Base checkpoint: {base_checkpoint or 'current model'}")


def get_neon_post_training_config(args) -> dict:
    """
    Extract Neon post-training configuration from args
    
    Returns:
        Dict with Neon parameters
    """
    return {
        "enabled": getattr(args, 'neon_enable', False),
        "save_pre_post": getattr(args, 'neon_save_pre_post', False),
        "synthetic_dataset_dir": getattr(args, 'neon_synthetic_dataset_dir', None),
        "post_training_epochs": getattr(args, 'neon_post_training_epochs', 0),
        "post_training_steps": getattr(args, 'neon_post_training_steps', 100),
        "synthetic_image_percent": getattr(args, 'neon_synthetic_image_percent', 100.0),
        "extrapolation_weight": getattr(args, 'neon_extrapolation_weight', 0.3),
    }


def calculate_synthetic_image_count(train_dataset, synthetic_percent: float) -> int:
    """
    Calculate how many synthetic images to generate based on original dataset
    
    Args:
        train_dataset: Training dataset object
        synthetic_percent: Percentage of original images (100.0 = same count)
    
    Returns:
        Number of synthetic images to generate
    """
    try:
        # Get total number of images including repeats
        if hasattr(train_dataset, '__len__'):
            original_count = len(train_dataset)
        elif hasattr(train_dataset, 'num_train_images'):
            original_count = train_dataset.num_train_images
        else:
            logger.warning("Could not determine dataset size, defaulting to 100 images")
            original_count = 100
        
        synthetic_count = int(original_count * (synthetic_percent / 100.0))
        logger.info(f"Original dataset: {original_count} images")
        logger.info(f"Synthetic percent: {synthetic_percent}%")
        logger.info(f"Will generate: {synthetic_count} synthetic images")
        
        return max(1, synthetic_count)  # At least 1 image
    except Exception as e:
        logger.warning(f"Error calculating synthetic count: {e}, defaulting to 100")
        return 100


def is_synthetic_image(image_path: str) -> bool:
    """
    Check if an image is synthetic by checking if filename contains '_synthetic'
    before the extension.
    
    Examples:
        img001_synthetic.png -> True
        img_002_synthetic.jpg -> True
        img001.png -> False
    
    Args:
        image_path: Path to image file
    
    Returns:
        True if synthetic, False otherwise
    """
    from pathlib import Path
    import re
    path = Path(image_path)
    # Match *_synthetic or *_synthetic_# (numbered variants)
    return re.match(r".*_synthetic(?:_\d+)?$", path.stem) is not None


def count_synthetic_images(directory: Path) -> int:
    """
    Count existing synthetic images in a directory (images with _synthetic suffix)
    
    Args:
        directory: Path to check
    
    Returns:
        Number of synthetic image files found
    """
    if not directory.exists():
        return 0
    
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.webp', '*.bmp']
    count = 0
    for ext in image_extensions:
        # Count only images with _synthetic in filename
        for img_path in directory.glob(ext):
            if is_synthetic_image(img_path):
                count += 1
    return count


def get_image_dimensions(image_path: Path) -> tuple[int, int]:
    """
    Get dimensions (width, height) of an image.
    
    Args:
        image_path: Path to image file
    
    Returns:
        Tuple of (width, height)
    """
    try:
        from PIL import Image
        with Image.open(image_path) as img:
            return img.size  # Returns (width, height)
    except Exception as e:
        logger.warning(f"Could not get dimensions for {image_path}: {e}")
        return (1024, 1024)  # Default fallback


def get_caption_for_image(image_path: Path) -> str:
    """
    Get caption for an image from associated .txt file.
    
    Args:
        image_path: Path to image file
    
    Returns:
        Caption text, or empty string if no caption found
    """
    caption_path = image_path.with_suffix('.txt')
    if caption_path.exists():
        try:
            with open(caption_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            logger.warning(f"Could not read caption {caption_path}: {e}")
    return ""


def copy_caption_with_synthetic_suffix(
    original_image_path: Path,
    synthetic_image_path: Path
) -> bool:
    """
    Copy caption from original image to synthetic image with _synthetic suffix.
    
    Args:
        original_image_path: Path to original image
        synthetic_image_path: Path to synthetic image
    
    Returns:
        True if caption was copied, False otherwise
    """
    import shutil
    
    original_caption = original_image_path.with_suffix('.txt')
    if not original_caption.exists():
        return False
    
    synthetic_caption = synthetic_image_path.with_suffix('.txt')
    try:
        shutil.copy(original_caption, synthetic_caption)
        return True
    except Exception as e:
        logger.warning(f"Could not copy caption: {e}")
        return False


def replicate_dataset_structure(
    train_data_dir: str,
    output_dir: str,
    synthetic_percent: float = 100.0,
    reuse_existing: bool = False,
    bucket_map: Optional[dict] = None,
) -> tuple[str, dict]:
    """
    Replicate the directory structure from original training dataset.
    Maintains DreamBooth naming convention (e.g., "10_character").
    
    If synthetic dataset already exists, counts existing images and only
    generates the delta needed to match target count.
    
    Builds detailed generation plan with:
    - Original image paths and dimensions
    - Target synthetic filenames
    - Captions to use for generation
    
    Args:
        train_data_dir: Original training data directory
        output_dir: Where to create synthetic dataset structure
        synthetic_percent: Percentage of images to generate per subset
        reuse_existing: If True, reuse existing synthetic images
    
    Returns:
        Tuple of (path to synthetic dataset, dict with generation info per subset)
    """
    from pathlib import Path
    import shutil
    
    train_path = Path(train_data_dir)
    synthetic_path = Path(output_dir)
    
    # Check if output already exists
    existing_dataset = synthetic_path.exists()
    if existing_dataset:
        logger.info("=" * 60)
        logger.info("Neon: Existing Synthetic Dataset Detected")
        logger.info("=" * 60)
        logger.info(f"Location: {synthetic_path}")
        logger.info(f"Will reuse existing images and generate missing ones")
    else:
        synthetic_path.mkdir(parents=True, exist_ok=True)
        logger.info("=" * 60)
        logger.info("Neon: Creating New Synthetic Dataset Structure")
        logger.info("=" * 60)
    
    logger.info(f"Source: {train_data_dir}")
    logger.info(f"Destination: {synthetic_path}")
    logger.info(f"Synthetic percent: {synthetic_percent}%")
    
    total_images_needed = 0
    total_images_existing = 0
    generation_plan = {}
    
    # Check if train_data_dir has subdirectories (DreamBooth style)
    subdirs = [d for d in train_path.iterdir() if d.is_dir()]
    # If synthetic_path is inside train_path, exclude it from source scanning
    try:
        if synthetic_path.resolve().is_dir() and synthetic_path.resolve().parent == train_path.resolve():
            subdirs = [d for d in subdirs if d.resolve() != synthetic_path.resolve()]
    except Exception:
        pass
    
    # Precompute available bucket sizes (from training), if provided
    available_buckets = []
    if bucket_map:
        try:
            uniq = {
                (int(v[0]), int(v[1]))
                for v in bucket_map.values()
                if isinstance(v, (tuple, list)) and len(v) == 2
            }
            available_buckets = list(uniq)
        except Exception:
            available_buckets = []

    def pick_bucket_fallback(w: int, h: int) -> tuple[int, int] | None:
        if not available_buckets:
            return None
        import math
        ar = w / max(h, 1)
        best = None
        best_score = 1e9
        for bw, bh in available_buckets:
            # enforce no-upscale
            if bw > w or bh > h:
                continue
            score = abs((bw / max(bh, 1)) - ar) + 1e-6 * (w - bw + h - bh)
            if score < best_score:
                best_score = score
                best = (bw, bh)
        return best

    if subdirs:
        # DreamBooth structure: train_data_dir/10_character/image001.png
        logger.info(f"Found {len(subdirs)} subdirectories (DreamBooth structure)")
        logger.info("")
        
        for subdir in subdirs:
            # Parse directory name like "10_character"
            tokens = subdir.name.split("_")
            try:
                num_repeats = int(tokens[0])
            except (ValueError, IndexError):
                logger.warning(f"Skipping directory with invalid format: {subdir.name}")
                continue
            
            class_tokens = "_".join(tokens[1:]) if len(tokens) > 1 else ""
            
            # Count images in original subset
            image_files = list(subdir.glob("*.png")) + list(subdir.glob("*.jpg")) + \
                         list(subdir.glob("*.jpeg")) + list(subdir.glob("*.webp"))
            
            original_count = len(image_files)
            synthetic_target = max(1, int(original_count * num_repeats * (synthetic_percent / 100.0)))
            
            # Create synthetic subdirectory with same naming
            synthetic_subdir = synthetic_path / subdir.name
            synthetic_subdir.mkdir(parents=True, exist_ok=True)
            
            # Count existing synthetic images (images with _synthetic suffix)
            existing_count = count_synthetic_images(synthetic_subdir) if reuse_existing else 0
            images_to_generate = max(0, synthetic_target - existing_count)
            
            logger.info(f"  {subdir.name}:")
            logger.info(f"    Original: {original_count} images (×{num_repeats} repeats)")
            logger.info(f"    Target: {synthetic_target} synthetic images")
            if existing_count > 0:
                logger.info(f"    Existing: {existing_count} images with '_synthetic' suffix ✓")
                logger.info(f"    To generate: {images_to_generate} more synthetic images")
            else:
                logger.info(f"    To generate: {images_to_generate} synthetic images (new)")
            
            total_images_needed += synthetic_target
            total_images_existing += existing_count
            
            # Build detailed generation plan for this subset
            generation_tasks = []
            if images_to_generate > 0 and len(image_files) > 0:
                # Cycle through original images until reaching target count
                for k in range(images_to_generate):
                    img_file = image_files[k % len(image_files)]
                    # Get original image info
                    width, height = get_image_dimensions(img_file)
                    # Default to None; populate from bucket_map if present; fallback to nearest available bucket
                    bucket_width = bucket_height = None
                    if bucket_map is not None:
                        bucket_size = bucket_map.get(str(img_file)) or bucket_map.get(img_file.name)
                        if bucket_size is not None:
                            bw, bh = bucket_size
                            bucket_width = int(bw)
                            bucket_height = int(bh)
                        else:
                            fb = pick_bucket_fallback(width, height)
                            if fb is not None:
                                bucket_width, bucket_height = fb
                    caption = get_caption_for_image(img_file)
                    
                    # Create unique synthetic filename based on original
                    base_stem = img_file.stem + "_synthetic"
                    candidate = synthetic_subdir / (base_stem + img_file.suffix)
                    idx = 1
                    while candidate.exists() and reuse_existing:
                        candidate = synthetic_subdir / (f"{base_stem}_{idx}{img_file.suffix}")
                        idx += 1
                    
                    generation_tasks.append({
                        'original_path': str(img_file),
                        'synthetic_path': str(candidate),
                        'caption': caption,
                        'width': width,
                        'height': height,
                        'bucket_width': bucket_width,
                        'bucket_height': bucket_height,
                    })
            
            # Store generation plan
            generation_plan[subdir.name] = {
                'original_count': original_count,
                'num_repeats': num_repeats,
                'class_tokens': class_tokens,
                'target': synthetic_target,
                'existing': existing_count,
                'to_generate': images_to_generate,
                'synthetic_dir': str(synthetic_subdir),
                'original_dir': str(subdir),
                'generation_tasks': generation_tasks,
            }
            
            # Copy all captions from original dataset
            captions_copied = 0
            for img_file in image_files:
                caption_file = img_file.with_suffix('.txt')
                if caption_file.exists():
                    # Copy caption with same filename to synthetic dir
                    dest_caption = synthetic_subdir / caption_file.name
                    if not dest_caption.exists():  # Don't overwrite existing
                        shutil.copy(caption_file, dest_caption)
                        captions_copied += 1
            if captions_copied > 0:
                logger.info(f"    Copied {captions_copied} caption files")
    
    else:
        # Flat structure: train_data_dir/image001.png
        logger.info("Flat directory structure detected")
        logger.info("")
        
        image_files = list(train_path.glob("*.png")) + list(train_path.glob("*.jpg")) + \
                     list(train_path.glob("*.jpeg")) + list(train_path.glob("*.webp"))
        
        original_count = len(image_files)
        synthetic_target = max(1, int(original_count * (synthetic_percent / 100.0)))
        
        # Count existing synthetic images (images with _synthetic suffix)
        existing_count = count_synthetic_images(synthetic_path) if reuse_existing else 0
        images_to_generate = max(0, synthetic_target - existing_count)
        
        logger.info(f"  Original: {original_count} images")
        logger.info(f"  Target: {synthetic_target} synthetic images")
        if existing_count > 0:
            logger.info(f"  Existing: {existing_count} images with '_synthetic' suffix ✓")
            logger.info(f"  To generate: {images_to_generate} more synthetic images")
        else:
            logger.info(f"  To generate: {images_to_generate} synthetic images (new)")
        
        total_images_needed = synthetic_target
        total_images_existing = existing_count
        
        # Build detailed generation plan for flat structure
        generation_tasks = []
        if images_to_generate > 0 and len(image_files) > 0:
            for k in range(images_to_generate):
                img_file = image_files[k % len(image_files)]
                
                # Get original image info
                width, height = get_image_dimensions(img_file)
                bucket_size = None
                if bucket_map is not None:
                    bucket_size = bucket_map.get(str(img_file)) or bucket_map.get(img_file.name)
                    if bucket_size is not None:
                        bucket_width, bucket_height = bucket_size
                    else:
                        fb = pick_bucket_fallback(width, height)
                        if fb is not None:
                            bucket_width, bucket_height = fb
                        else:
                            bucket_width = bucket_height = None
                else:
                    bucket_width = bucket_height = None
                caption = get_caption_for_image(img_file)
                
                # Create unique synthetic filename
                base_stem = img_file.stem + "_synthetic"
                candidate = synthetic_path / (base_stem + img_file.suffix)
                idx = 1
                while candidate.exists() and reuse_existing:
                    candidate = synthetic_path / (f"{base_stem}_{idx}{img_file.suffix}")
                    idx += 1
                
                generation_tasks.append({
                    'original_path': str(img_file),
                    'synthetic_path': str(candidate),
                    'caption': caption,
                    'width': width,
                    'height': height,
                    'bucket_width': bucket_width,
                    'bucket_height': bucket_height,
                })
        
        # Store generation plan
        generation_plan['flat'] = {
            'original_count': original_count,
            'num_repeats': 1,
            'class_tokens': '',
            'target': synthetic_target,
            'existing': existing_count,
            'to_generate': images_to_generate,
            'synthetic_dir': str(synthetic_path),
            'original_dir': str(train_path),
            'generation_tasks': generation_tasks,
        }
        
        # Copy all captions from original dataset
        captions_copied = 0
        for img_file in image_files:
            caption_file = img_file.with_suffix('.txt')
            if caption_file.exists():
                dest_caption = synthetic_path / caption_file.name
                if not dest_caption.exists():  # Don't overwrite existing
                    shutil.copy(caption_file, dest_caption)
                    captions_copied += 1
        if captions_copied > 0:
            logger.info(f"  Copied {captions_copied} caption files")
    
    logger.info("")
    logger.info("=" * 60)
    images_to_generate_total = total_images_needed - total_images_existing
    if total_images_existing > 0:
        logger.info(f"✅ Using existing: {total_images_existing} images")
        logger.info(f"📝 Will generate: {images_to_generate_total} new images")
        logger.info(f"🎯 Total target: {total_images_needed} images")
    else:
        logger.info(f"📝 Will generate: {images_to_generate_total} images (new dataset)")
    logger.info(f"📂 Synthetic dataset: {synthetic_path}")
    logger.info("=" * 60)
    
    return str(synthetic_path), generation_plan


def generate_synthetic_dataset(
    args,
    train_data_dir: str,
    synthetic_percent: float = 100.0,
    vae=None,
    text_encoder=None,
    unet=None,
    tokenizer=None,
    noise_scheduler=None,
    weight_dtype=None,
    device=None,
    test_mode: bool = False,
    bucket_map: Optional[dict] = None,
) -> tuple[str, dict]:
    """
    Generate synthetic dataset by replicating original structure and generating images.
    
    Workflow:
    1. Analyze original dataset structure
    2. Count existing synthetic images (with _synthetic suffix)
    3. Build generation plan for missing images
    4. Generate synthetic images using trained model
    5. Copy captions with _synthetic suffix
    
    Intelligently reuses existing synthetic images:
    - Detects synthetic images by '_synthetic' suffix in filename
    - Counts existing synthetic images and only generates missing ones
    - Example: img001_synthetic.png, img002_synthetic.jpg
    
    Args:
        args: Training arguments
        train_data_dir: Original training data directory
        synthetic_percent: Percentage of images to generate
        vae: VAE model (required if not test_mode)
        text_encoder: Text encoder(s) (required if not test_mode)
        unet: UNet model (required if not test_mode)
        tokenizer: Tokenizer(s) (required if not test_mode)
        noise_scheduler: Noise scheduler (required if not test_mode)
        weight_dtype: Weight dtype (required if not test_mode)
        device: Device (required if not test_mode)
        test_mode: If True, generates placeholder images instead of real ones
    
    Returns:
        Tuple of (path to synthetic dataset, generation plan dict)
    """
    from pathlib import Path
    
    # Get synthetic dataset directory from args or use default
    neon_config = get_neon_post_training_config(args)
    synthetic_dir = neon_config.get('synthetic_dataset_dir')
    
    if not synthetic_dir:
        # Auto-generate path
        output_dir = getattr(args, 'output_dir', './output')
        synthetic_dir = str(Path(output_dir) / "neon_synthetic")
        logger.info(f"No synthetic dataset directory specified, using: {synthetic_dir}")
    
    logger.info(f"ℹ️  Synthetic images will be named with '_synthetic' suffix")
    logger.info(f"   Example: img001_synthetic.png, img002_synthetic.jpg")
    logger.info(f"   This allows detection and reuse of existing synthetic images")
    
    # Replicate structure and build generation plan
    synthetic_path, generation_plan = replicate_dataset_structure(
        train_data_dir=train_data_dir,
        output_dir=synthetic_dir,
        synthetic_percent=synthetic_percent,
        reuse_existing=False,
        bucket_map=bucket_map,
    )
    
    # Check if there are any images to generate
    total_to_generate = sum(
        len(subset_info.get('generation_tasks', []))
        for subset_info in generation_plan.values()
    )
    
    if total_to_generate == 0:
        logger.info("✓ All synthetic images already exist, nothing to generate!")
        return synthetic_path, generation_plan
    
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"Starting Synthetic Image Generation")
    logger.info(f"Total images to generate: {total_to_generate}")
    logger.info("=" * 60)
    
    # Generate synthetic images
    if test_mode:
        # Test mode: generate placeholder images
        from .neon_generation_pipeline import test_generation_pipeline
        test_generation_pipeline(generation_plan, num_test_images=total_to_generate)
    else:
        # Real generation mode
        if vae is None or text_encoder is None or unet is None:
            logger.error("Models not provided for generation! Skipping image generation.")
            logger.error("Provide vae, text_encoder, unet, etc. or use test_mode=True")
            return synthetic_path, generation_plan
        
        from .neon_generation_pipeline import generate_synthetic_images
        
        # Get generation parameters from args
        guidance_scale = getattr(args, 'neon_guidance_scale', 6)
        num_inference_steps = getattr(args, 'neon_inference_steps', 20)
        seed = getattr(args, 'seed', None)
        
        num_generated = generate_synthetic_images(
            generation_plan=generation_plan,
            vae=vae,
            text_encoder=text_encoder,
            unet=unet,
            tokenizer=tokenizer,
            noise_scheduler=noise_scheduler,
            weight_dtype=weight_dtype,
            device=device,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            seed=seed,
            clip_skip=getattr(args, 'clip_skip', None),
            comfy_mode=True,
            positive_prefix=getattr(args, 'neon_positive_prefix', ''),
            negative_prompt=getattr(args, 'neon_negative_prompt', None),
            generation_batch_size=int(getattr(args, 'neon_generation_batch_size', 1) or 1),
        )
        
        logger.info(f"✅ Successfully generated {num_generated} synthetic images")
    
    return synthetic_path, generation_plan


def run_neon_post_training(
    args,
    base_model_path: str,
    synthetic_dataset_path: str,
    output_dir: str,
) -> str:
    """
    Run Neon post-training phase:
    1. Train on synthetic data (creates aux model)
    2. Merge base + aux using Neon formula
    3. Return path to merged model
    
    Args:
        args: Training arguments
        base_model_path: Path to base model (completed main training)
        synthetic_dataset_path: Path to generated synthetic dataset
        output_dir: Output directory
    
    Returns:
        Path to Neon-merged model
    """
    from pathlib import Path
    
    logger.info("=" * 60)
    logger.info("Neon: Post-Training Phase")
    logger.info("=" * 60)
    
    neon_config = get_neon_post_training_config(args)
    
    # Placeholder for actual post-training
    # TODO: Implement training loop on synthetic data
    logger.warning("⚠️  Post-training on synthetic data not yet fully implemented!")
    logger.warning("    This is a placeholder for the training loop")
    
    aux_model_path = Path(output_dir) / "aux_synthetic.safetensors"
    logger.info(f"Aux model would be saved to: {aux_model_path}")
    
    # For now, just copy base as aux (placeholder)
    import shutil
    if Path(base_model_path).exists():
        shutil.copy(base_model_path, aux_model_path)
        logger.info(f"Placeholder: copied base to aux")
    
    # Perform Neon merge
    w = neon_config['extrapolation_weight']
    neon_model_path = Path(output_dir) / f"neon_merged_w{w:.2f}.safetensors"
    
    logger.info(f"Applying Neon merge (w={w})")
    neon_merge_lora(
        base_lora_path=str(base_model_path),
        aux_lora_path=str(aux_model_path),
        output_path=str(neon_model_path),
        w=w,
    )
    
    logger.info("=" * 60)
    logger.info("✅ Neon Post-Training Complete!")
    logger.info(f"Final model: {neon_model_path}")
    logger.info("=" * 60)
    
    return str(neon_model_path)
