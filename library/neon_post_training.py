"""
Neon Post-Training Loop

Runs additional training on synthetic dataset to create auxiliary model.
"""

import torch
from pathlib import Path
import logging

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
    
    for epoch in range(epochs_to_train):
        for batch in train_dataloader:
            if num_steps > 0 and current_step >= num_steps:
                break
            
            with accelerator.accumulate(network):
                # Standard diffusion training step
                # This is a simplified version - adapt based on your actual training loop
                
                # Get pixel values and convert to latents
                pixel_values = batch["pixel_values"].to(accelerator.device, dtype=vae.dtype)
                if pixel_values.ndim == 3:
                    pixel_values = pixel_values.unsqueeze(0)
                # already normalized to [-1,1] by transform

                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * (vae.config.scaling_factor if hasattr(vae.config, "scaling_factor") else 0.18215)

                latents = latents.to(dtype=weight_dtype)
                
                # Encode text (apply optional clip_skip to match training strategy)
                clip_skip = getattr(args, 'clip_skip', None)
                if isinstance(text_encoder, (list, tuple)):
                    hs = []
                    for i, encoder in enumerate(text_encoder):
                        input_ids = batch[f"input_ids_{i}"].to(accelerator.device)
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
                    input_ids = batch["input_ids"].to(accelerator.device)
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
                
                # Sample noise
                noise = torch.randn_like(latents)
                
                # Sample timestep
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                )
                timesteps = timesteps.long()
                
                # Add noise to latents
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Predict noise
                noise_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states,
                ).sample
                
                # Compute loss
                loss = torch.nn.functional.mse_loss(
                    noise_pred.float(),
                    noise.float(),
                    reduction="mean"
                )
                
                # Backward pass
                accelerator.backward(loss)
                
                # Clip gradients if specified
                if hasattr(args, 'max_grad_norm') and args.max_grad_norm is not None:
                    accelerator.clip_grad_norm_(network.parameters(), args.max_grad_norm)
                
                # Optimizer step
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Update progress
            current_step += 1
            global_step += 1
            progress_bar.update(1)
            progress_bar.set_postfix({"loss": loss.item()})
            
            if num_steps > 0 and current_step >= num_steps:
                break
    
    progress_bar.close()
    
    logger.info(f"✓ Post-training complete ({current_step} steps)")
    
    return network


def create_synthetic_dataloader(
    synthetic_dataset_path: str,
    batch_size: int,
    tokenizers,
    vae,
    resolution: int = 1024,
    accelerator=None,
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
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Use 0 for simplicity
        drop_last=True,
    )
    
    if accelerator is not None:
        dataloader = accelerator.prepare(dataloader)
    
    return dataloader
