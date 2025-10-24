"""
Neon Post-Training Loop

Runs additional training on synthetic dataset to create auxiliary model.
"""

import torch
from pathlib import Path
import logging
import os
from library.device_utils import clean_memory_on_device
from library import train_util, sdxl_train_util
from library.custom_train_functions import (
    add_v_prediction_like_loss,
    apply_debiased_estimation,
    apply_snr_weight,
    fix_noise_scheduler_betas_for_zero_terminal_snr,
    prepare_scheduler_for_custom_training,
    scale_v_prediction_loss_like_noise_prediction,
)

logger = logging.getLogger(__name__)


def run_post_training_loop(
    args,
    accelerator,
    network,
    optimizer,
    train_dataloader,
    lr_scheduler,
    num_steps: int,
    num_epochs: int,
    vae,
    text_encoder,
    unet,
    noise_scheduler,
    weight_dtype,
    global_step: int = 0,
):
    """
    Run post-training loop on synthetic dataset.
    
    This creates the "auxiliary" model (θ_aux) for Neon merge.
    
    Args:
        args: Training arguments
        accelerator: Accelerate accelerator
        network: Network to train (LoRA)
        optimizer: Optimizer
        train_dataloader: DataLoader for synthetic dataset
        lr_scheduler: Learning rate scheduler
        num_steps: Number of steps to train (if > 0, overrides epochs)
        num_epochs: Number of epochs to train (used if num_steps == 0)
        vae: VAE model
        text_encoder: Text encoder(s)
        unet: UNet model
        noise_scheduler: Noise scheduler
        weight_dtype: Weight dtype
        global_step: Starting global step
    
    Returns:
        Trained network
    """
    from tqdm import tqdm
    
    # Memory strategy
    logger.info("Preparing models for post-training (optimizing memory)...")
    device = accelerator.device
    use_low_vram = bool(getattr(args, "lowram", False) or getattr(args, "neon_low_vram", False))

    # Align scheduler behavior with primary training scripts
    prepare_scheduler_for_custom_training(noise_scheduler, device)
    if getattr(args, "zero_terminal_snr", False):
        fix_noise_scheduler_betas_for_zero_terminal_snr(noise_scheduler)

    # Save original device placements
    unet_device = unet.device
    text_encoder_devices = []
    if isinstance(text_encoder, (list, tuple)):
        for te in text_encoder:
            text_encoder_devices.append(te.device)
    else:
        text_encoder_devices.append(text_encoder.device)

    # Place models based on memory mode
    if use_low_vram:
        # Low-VRAM path: keep UNet/Text Encoders on CPU until needed, VAE on GPU for encoding
        if isinstance(text_encoder, (list, tuple)):
            for te in text_encoder:
                te.to("cpu")
        else:
            text_encoder.to("cpu")
        unet.to("cpu")
        clean_memory_on_device(device)

        vae.requires_grad_(False)
        vae.eval()
        vae.to(device, dtype=vae.dtype)
    else:
        # High-VRAM path: keep everything on device to avoid costly shuffles
        vae.requires_grad_(False)
        vae.eval()
        vae.to(device, dtype=vae.dtype)
        if isinstance(text_encoder, (list, tuple)):
            for te in text_encoder:
                te.to(device)
        else:
            text_encoder.to(device)
        unet.to(device)
    
    network.train()
    
    # Determine training length
    if num_steps > 0:
        total_steps = num_steps
        epochs_to_train = 1  # Just run through dataloader once
        logger.info(f"Post-training for {total_steps} steps")
    else:
        total_steps = len(train_dataloader) * num_epochs
        epochs_to_train = num_epochs
        logger.info(f"Post-training for {num_epochs} epochs ({total_steps} steps)")
    
    progress_bar = tqdm(
        total=total_steps,
        desc="Neon post-training",
        disable=not accelerator.is_main_process,
    )
    
    current_step = 0
    
    dataset_obj = getattr(train_dataloader, "dataset", None)
    dataset_tokenizers = getattr(dataset_obj, "tokenizers", None)

    for epoch in range(epochs_to_train):
        for batch in train_dataloader:
            if num_steps > 0 and current_step >= num_steps:
                break
            
            with accelerator.accumulate(network):
                # Standard diffusion training step
                # This is a simplified version - adapt based on your actual training loop
                
                # Get pixel values and convert to latents
                pixel_values = batch["pixel_values"].to(device, dtype=vae.dtype, non_blocking=True)
                if pixel_values.ndim == 3:
                    pixel_values = pixel_values.unsqueeze(0)
                # already normalized to [-1,1] by transform

                # Process VAE encoding in smaller batches to save memory
                vae_batch_size = getattr(args, 'vae_batch_size', 1) or 1  # Default to 1 for safety
                with torch.no_grad():
                    if pixel_values.shape[0] <= vae_batch_size:
                        latents = vae.encode(pixel_values).latent_dist.sample()
                    else:
                        # Process in chunks
                        latent_chunks = []
                        for i in range(0, pixel_values.shape[0], vae_batch_size):
                            chunk = pixel_values[i:i + vae_batch_size]
                            latent_chunk = vae.encode(chunk).latent_dist.sample()
                            latent_chunks.append(latent_chunk)
                        latents = torch.cat(latent_chunks, dim=0)
                    
                    latents = latents * (vae.config.scaling_factor if hasattr(vae.config, "scaling_factor") else 0.18215)

                latents = latents.to(dtype=weight_dtype)
                
                # Free memory more aggressively only when operating in low-VRAM mode
                if use_low_vram:
                    vae.to("cpu")
                    clean_memory_on_device(device)
                else:
                    # Keep primary models resident on device for smoother allocator usage
                    if isinstance(text_encoder, (list, tuple)):
                        for te in text_encoder:
                            te.to(device)
                    else:
                        text_encoder.to(device)
                    unet.to(device)
                    unet.train()
                    network.train()
                    if hasattr(optimizer, "train"):
                        optimizer.train()
                
                vector_embeddings = None

                # Encode text (apply optional clip_skip to match training strategy)
                clip_skip = getattr(args, 'clip_skip', None)
                if isinstance(text_encoder, (list, tuple)):
                    if len(text_encoder) == 2 and dataset_tokenizers is not None:
                        input_ids0 = batch["input_ids_0"].to(accelerator.device, non_blocking=True)
                        input_ids1 = batch["input_ids_1"].to(accelerator.device, non_blocking=True)
                        max_token_length = getattr(args, "max_token_length", None)
                        with torch.no_grad():
                            hidden_states1, hidden_states2, pool2 = train_util.get_hidden_states_sdxl(
                                max_token_length,
                                input_ids0,
                                input_ids1,
                                dataset_tokenizers[0],
                                dataset_tokenizers[1],
                                text_encoder[0],
                                text_encoder[1],
                                weight_dtype if getattr(args, "full_fp16", False) else None,
                                accelerator=accelerator,
                            )

                        encoder_hidden_states = torch.cat([hidden_states1, hidden_states2], dim=-1)

                        bsz = encoder_hidden_states.shape[0]
                        height, width = pixel_values.shape[-2], pixel_values.shape[-1]
                        orig_size = torch.tensor([height, width], device=accelerator.device, dtype=torch.int64).repeat(bsz, 1)
                        crop_size = torch.zeros_like(orig_size)
                        target_size = orig_size.clone()
                        size_embeddings = sdxl_train_util.get_size_embeddings(
                            orig_size,
                            crop_size,
                            target_size,
                            accelerator.device,
                        ).to(weight_dtype)

                        pool2 = pool2.to(accelerator.device, dtype=weight_dtype)
                        if pool2.shape[0] != bsz:
                            repeat_factor = (bsz + pool2.shape[0] - 1) // pool2.shape[0]
                            pool2 = pool2.repeat_interleave(repeat_factor, dim=0)[:bsz]

                        vector_embeddings = torch.cat([pool2, size_embeddings], dim=1).to(weight_dtype)
                    else:
                        hs = []
                        for i, encoder in enumerate(text_encoder):
                            input_ids = batch[f"input_ids_{i}"].to(accelerator.device, non_blocking=True)
                            if clip_skip is None:
                                out = encoder(input_ids)[0]
                            else:
                                out_dict = encoder(input_ids, output_hidden_states=True, return_dict=True)
                                idx = -clip_skip if clip_skip and clip_skip > 0 else -2
                                out = out_dict["hidden_states"][idx]
                                try:
                                    tm = getattr(encoder, "text_model", None)
                                    if tm is not None and hasattr(tm, "final_layer_norm"):
                                        out = tm.final_layer_norm(out)
                                except Exception:
                                    pass
                            hs.append(out)
                        encoder_hidden_states = torch.cat(hs, dim=-1)
                else:
                    input_ids = batch["input_ids"].to(accelerator.device, non_blocking=True)
                    if clip_skip is None:
                        encoder_hidden_states = text_encoder(input_ids)[0]
                    else:
                        out_dict = text_encoder(input_ids, output_hidden_states=True, return_dict=True)
                        idx = -clip_skip if clip_skip and clip_skip > 0 else -2
                        encoder_hidden_states = out_dict["hidden_states"][idx]
                        try:
                            tm = getattr(text_encoder, "text_model", None)
                            if tm is not None and hasattr(tm, "final_layer_norm"):
                                encoder_hidden_states = tm.final_layer_norm(encoder_hidden_states)
                        except Exception:
                            pass

                # pixel_values are no longer needed past this point; drop the reference to keep allocator steady
                del pixel_values

                # Ensure embeddings use compute dtype to save VRAM
                encoder_hidden_states = encoder_hidden_states.to(accelerator.device, dtype=weight_dtype)

                # Sample noise, timesteps, and noisy latents using shared utility (handles noise offset, IP noise, etc.)
                noise, noisy_latents, timesteps = train_util.get_noise_noisy_latents_and_timesteps(
                    args, noise_scheduler, latents
                )
                
                # Predict noise (mixed precision)
                with accelerator.autocast():
                    if vector_embeddings is not None:
                        noise_pred_out = unet(
                            noisy_latents,
                            timesteps,
                            encoder_hidden_states,
                            vector_embeddings,
                        )
                    else:
                        noise_pred_out = unet(
                            noisy_latents,
                            timesteps,
                            encoder_hidden_states,
                        )

                noise_pred = noise_pred_out.sample if hasattr(noise_pred_out, "sample") else noise_pred_out
                
                # Compute loss with repo-standard options (v-pred, huber, SNR weighting, etc.)
                huber_c = train_util.get_huber_threshold_if_needed(args, timesteps, noise_scheduler)
                if getattr(args, "v_parameterization", False):
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    target = noise

                loss = train_util.conditional_loss(
                    noise_pred.float(),
                    target.float(),
                    getattr(args, "loss_type", "l2"),
                    "none",
                    huber_c,
                )

                # Reduce to per-sample loss
                if loss.ndim > 1:
                    reduce_dims = tuple(range(1, loss.ndim))
                    loss = loss.mean(dim=reduce_dims)

                # Apply optional loss transforms
                min_snr_gamma = getattr(args, "min_snr_gamma", None)
                if min_snr_gamma is not None:
                    loss = apply_snr_weight(
                        loss,
                        timesteps,
                        noise_scheduler,
                        min_snr_gamma,
                        getattr(args, "v_parameterization", False),
                    )

                if getattr(args, "scale_v_pred_loss_like_noise_pred", False):
                    loss = scale_v_prediction_loss_like_noise_prediction(loss, timesteps, noise_scheduler)

                v_pred_like = getattr(args, "v_pred_like_loss", None)
                if v_pred_like is not None:
                    loss = add_v_prediction_like_loss(loss, timesteps, noise_scheduler, v_pred_like)

                if getattr(args, "debiased_estimation_loss", False):
                    loss = apply_debiased_estimation(
                        loss,
                        timesteps,
                        noise_scheduler,
                        getattr(args, "v_parameterization", False),
                    )

                loss = loss.mean()
                
                # Backward pass
                accelerator.backward(loss)
                
                # Clip gradients if specified
                if hasattr(args, 'max_grad_norm') and args.max_grad_norm is not None:
                    accelerator.clip_grad_norm_(network.parameters(), args.max_grad_norm)
                
                # Optimizer step
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            
            # In low-VRAM mode, move models back to CPU after the step and ready VAE for next encode
            if use_low_vram:
                if isinstance(text_encoder, (list, tuple)):
                    for te in text_encoder:
                        te.to("cpu")
                else:
                    text_encoder.to("cpu")
                unet.to("cpu")
                clean_memory_on_device(device)

                # Move VAE back to GPU for next iteration
                vae.to(device, dtype=vae.dtype)
            
            # Update progress
            current_step += 1
            global_step += 1
            progress_bar.update(1)
            progress_bar.set_postfix({"loss": loss.item()})
            
            if num_steps > 0 and current_step >= num_steps:
                break
    
    progress_bar.close()
    
    # Restore original device placement
    logger.info("Restoring model devices after post-training...")
    vae.to("cpu")
    if isinstance(text_encoder, (list, tuple)):
        for i, te in enumerate(text_encoder):
            te.to(text_encoder_devices[i])
    else:
        text_encoder.to(text_encoder_devices[0])
    unet.to(unet_device)
    clean_memory_on_device(accelerator.device)
    
    logger.info(f"✓ Post-training complete ({current_step} steps)")
    
    return network


def create_synthetic_dataloader(
    synthetic_dataset_path: str,
    batch_size: int,
    tokenizers,
    vae,
    resolution: int = 1024,
    accelerator=None,
    args=None,
):
    """
    Create a dataloader for synthetic dataset.
    
    This is a simplified version - you may need to adapt based on your
    actual dataset loading infrastructure.
    
    Args:
        synthetic_dataset_path: Path to synthetic dataset
        batch_size: Batch size
        tokenizer: Tokenizer for captions
        vae: VAE for encoding images
        resolution: Target resolution
        accelerator: Accelerate accelerator
    
    Returns:
        DataLoader for synthetic dataset
    """
    from torch.utils.data import DataLoader
    from PIL import Image
    import torchvision.transforms as transforms
    import torch.utils.data as data
    
    class SyntheticDataset(data.Dataset):
        def __init__(self, dataset_path, tokenizers, resolution):
            self.dataset_path = Path(dataset_path)
            if isinstance(tokenizers, (list, tuple)):
                self.tokenizers = list(tokenizers)
            else:
                self.tokenizers = [tokenizers]
            self.resolution = resolution
            
            # Find all synthetic images
            self.image_files = []
            for ext in ['.png', '.jpg', '.jpeg', '.webp']:
                self.image_files.extend(list(self.dataset_path.rglob(f'*_synthetic{ext}')))
            
            logger.info(f"Found {len(self.image_files)} synthetic images")
            
            self.transform = transforms.Compose([
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # map [0,1] -> [-1,1] per channel
            ])
        
        def __len__(self):
            return len(self.image_files)
        
        def __getitem__(self, idx):
            img_path = self.image_files[idx]
            
            # Load image
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
            
            # Load caption
            caption_path = img_path.with_suffix('.txt')
            if caption_path.exists():
                with open(caption_path, 'r', encoding='utf-8') as f:
                    caption = f.read().strip()
            else:
                caption = ""
            
            # Tokenize caption
            data = {
                "pixel_values": image,
                "caption": caption,
            }

            if len(self.tokenizers) == 1:
                tokens = self.tokenizers[0](
                    caption,
                    padding="max_length",
                    max_length=self.tokenizers[0].model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                data["input_ids"] = tokens.input_ids[0]
            else:
                for idx, tok in enumerate(self.tokenizers):
                    tokens = tok(
                        caption,
                        padding="max_length",
                        max_length=tok.model_max_length,
                        truncation=True,
                        return_tensors="pt",
                    )
                    data[f"input_ids_{idx}"] = tokens.input_ids[0]

            return data
    
    dataset = SyntheticDataset(synthetic_dataset_path, tokenizers, resolution)

    # Configure DataLoader workers and perf flags
    n_workers = 0
    persistent_workers = False
    pin_memory = False
    prefetch_kwargs = {}
    if args is not None:
        try:
            cpu_count = os.cpu_count() or 1
            n_workers = max(0, min(getattr(args, 'max_data_loader_n_workers', 0) or 0, cpu_count))
        except Exception:
            n_workers = 0
        persistent_workers = bool(getattr(args, 'persistent_data_loader_workers', False) and n_workers > 0)
        # pin_memory helps when transferring to CUDA/XPU
        if accelerator is not None and hasattr(accelerator, 'device'):
            pin_memory = accelerator.device.type in ("cuda", "xpu")
        # prefetch_factor only valid with workers > 0
        pf = getattr(args, 'prefetch_factor', None)
        if pf is not None and n_workers > 0:
            prefetch_kwargs['prefetch_factor'] = int(pf)

    if n_workers > 0:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=n_workers,
            persistent_workers=persistent_workers,
            pin_memory=pin_memory,
            drop_last=True,
            **prefetch_kwargs,
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            drop_last=True,
        )
    
    if accelerator is not None:
        dataloader = accelerator.prepare(dataloader)
    
    return dataloader
