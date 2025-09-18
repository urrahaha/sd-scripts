# SDXL full finetuning with Flow Matching (Diff2Flow-style)
# DreamBooth/caption training variant adapted from sdxl_train.py

import argparse
import math
import os
import toml
import time
import sys
import faulthandler
from multiprocessing import Value
from typing import List, Optional, Any

import torch
from tqdm import tqdm

from library.device_utils import init_ipex, clean_memory_on_device

init_ipex()

from accelerate.utils import set_seed
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.training_utils import compute_density_for_timestep_sampling
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler, record_function

from library import deepspeed_utils, sdxl_model_util, strategy_base, strategy_sd, strategy_sdxl
import library.train_util as train_util
from library.utils import setup_logging, add_logging_arguments

setup_logging()
import logging

logger = logging.getLogger(__name__)

import library.config_util as config_util
import library.sdxl_train_util as sdxl_train_util
from library.config_util import (
    ConfigSanitizer,
    BlueprintGenerator,
)
import library.custom_train_functions as custom_train_functions
from library.custom_train_functions import apply_masked_loss
from library.sdxl_original_unet import SdxlUNet2DConditionModel


UNET_NUM_BLOCKS_FOR_BLOCK_LR = 23
_warned_min_snr_gamma = False
_warned_vpred_flags = False


def get_block_params_to_optimize(unet: SdxlUNet2DConditionModel, block_lrs: List[float]) -> List[dict]:
    block_params = [[] for _ in range(len(block_lrs))]

    for i, (name, param) in enumerate(unet.named_parameters()):
        if name.startswith("time_embed.") or name.startswith("label_emb."):
            block_index = 0  # 0
        elif name.startswith("input_blocks."):  # 1-9
            block_index = 1 + int(name.split(".")[1])
        elif name.startswith("middle_block."):  # 10-12
            block_index = 10 + int(name.split(".")[1])
        elif name.startswith("output_blocks."):  # 13-21
            block_index = 13 + int(name.split(".")[1])
        elif name.startswith("out."):  # 22
            block_index = 22
        else:
            raise ValueError(f"unexpected parameter name: {name}")

        block_params[block_index].append(param)

    params_to_optimize = []
    for i, params in enumerate(block_params):
        if block_lrs[i] == 0:  # 0のときは学習しない do not optimize when lr is 0
            continue
        params_to_optimize.append({"params": params, "lr": block_lrs[i]})

    return params_to_optimize


def append_block_lr_to_logs(block_lrs, logs, lr_scheduler, optimizer_type):
    names = []
    block_index = 0
    while block_index < UNET_NUM_BLOCKS_FOR_BLOCK_LR + 2:
        if block_index < UNET_NUM_BLOCKS_FOR_BLOCK_LR:
            if block_lrs[block_index] == 0:
                block_index += 1
                continue
            names.append(f"block{block_index}")
        elif block_index == UNET_NUM_BLOCKS_FOR_BLOCK_LR:
            names.append("text_encoder1")
        elif block_index == UNET_NUM_BLOCKS_FOR_BLOCK_LR + 1:
            names.append("text_encoder2")

        block_index += 1

    train_util.append_lr_to_logs_with_names(logs, lr_scheduler, optimizer_type, names)


# ----- Diff2Flow helpers -----

def _get_sigmas(
    noise_scheduler: FlowMatchEulerDiscreteScheduler,
    device: torch.device,
    timesteps: torch.Tensor,
    n_dim: int = 4,
    dtype: torch.dtype = torch.float32,
):
    """
    Vectorized lookup of sigma values for given timesteps from FlowMatch scheduler.
    Matches logic used in `sd-scripts/mosaic_train.py` and LoRA FM trainer.
    """
    sigmas = noise_scheduler.sigmas.to(device=device, dtype=dtype, non_blocking=True)
    sched_ts = noise_scheduler.timesteps.to(device, non_blocking=True)

    # scheduler timesteps are descending; flip to ascending for searchsorted
    asc_ts = torch.flip(sched_ts, dims=(0,))
    idx = torch.searchsorted(asc_ts, timesteps.to(device))
    idx = (len(sched_ts) - 1) - idx

    sigma = sigmas.index_select(0, idx)
    extra = (1,) * max(n_dim - sigma.ndim, 0)
    return sigma.reshape(*sigma.shape, *extra)


class Diff2FlowWrapper:
    """
    Diff2Flow wrapper that bridges diffusion and flow matching paradigms.
    Enables knowledge transfer from pre-trained diffusion models to FM.
    """
    
    def __init__(self, unet, noise_scheduler, parameterization="v"):
        self.unet = unet
        self.noise_scheduler = noise_scheduler
        self.parameterization = parameterization  # "v" or "eps"
        self.num_timesteps = 1000
        
        # Initialize diffusion schedule (SDXL uses cosine-like schedule)
        self._setup_diffusion_schedule()
    
    def _setup_diffusion_schedule(self):
        """Setup diffusion schedule for trajectory alignment"""
        import numpy as np
        from functools import partial
        
        # SDXL-style schedule
        linear_start = 0.00085
        linear_end = 0.0120
        
        # Create beta schedule
        betas = self._make_beta_schedule(
            "scaled_linear", self.num_timesteps, linear_start, linear_end
        )
        
        # Enforce zero terminal SNR for better FM alignment
        betas = self._enforce_zero_terminal_snr(betas)
        
        # Numerical stability: clamp arrays to valid ranges to avoid invalid sqrt
        eps = 1e-12
        betas = np.clip(betas, eps, 1.0 - eps)
        alphas = 1.0 - betas
        alphas = np.clip(alphas, eps, 1.0 - eps)
        
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod = np.clip(alphas_cumprod, eps, 1.0)
        alphas_cumprod_full = np.append(1.0, alphas_cumprod)
        alphas_cumprod_full = np.clip(alphas_cumprod_full, eps, 1.0)
        
        to_torch = partial(torch.tensor, dtype=torch.float32)
        
        # Register buffers
        self.register_buffer = lambda name, tensor: setattr(self, name, tensor)
        
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_full', to_torch(alphas_cumprod_full))
        
        # Precompute stable square-roots and reciprocals
        sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = np.sqrt(np.clip(1.0 - alphas_cumprod, 0.0, 1.0))
        sqrt_alphas_cumprod_full = np.sqrt(alphas_cumprod_full)
        sqrt_one_minus_alphas_cumprod_full = np.sqrt(np.clip(1.0 - alphas_cumprod_full, 0.0, 1.0))
        
        inv_alphas_cumprod = 1.0 / np.clip(alphas_cumprod, eps, 1.0)
        sqrt_recip_alphas_cumprod = np.sqrt(inv_alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = np.sqrt(np.clip(inv_alphas_cumprod - 1.0, 0.0, None))
        
        self.register_buffer('sqrt_alphas_cumprod', to_torch(sqrt_alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(sqrt_one_minus_alphas_cumprod))
        self.register_buffer('sqrt_alphas_cumprod_full', to_torch(sqrt_alphas_cumprod_full))
        self.register_buffer('sqrt_one_minus_alphas_cumprod_full', to_torch(sqrt_one_minus_alphas_cumprod_full))
        
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(sqrt_recip_alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(sqrt_recipm1_alphas_cumprod))
        
        # Rectified flow alignment coefficients
        scale = self.sqrt_alphas_cumprod_full + self.sqrt_one_minus_alphas_cumprod_full
        self.register_buffer('rectified_alphas_cumprod_full', 
                           self.sqrt_alphas_cumprod_full / scale)
        self.register_buffer('rectified_sqrt_alphas_cumprod_full', 
                           self.sqrt_one_minus_alphas_cumprod_full / scale)
    
    def _make_beta_schedule(self, schedule, n_timestep, linear_start=1e-4, linear_end=2e-2):
        """Create beta schedule"""
        import numpy as np
        
        if schedule == "linear":
            betas = np.linspace(linear_start**0.5, linear_end**0.5, n_timestep, dtype=np.float64) ** 2
        elif schedule == "scaled_linear":
            betas = np.linspace(linear_start**0.5, linear_end**0.5, n_timestep, dtype=np.float64) ** 2
        else:
            raise ValueError(f"Unknown beta schedule: {schedule}")
        
        return betas
    
    def _enforce_zero_terminal_snr(self, betas):
        """Enforce zero terminal SNR"""
        import numpy as np
        
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        
        # Rescale to ensure zero terminal SNR
        alphas_cumprod_final = alphas_cumprod[-1]
        if alphas_cumprod_final > 0:
            alphas_cumprod = alphas_cumprod / alphas_cumprod_final
            alphas = alphas_cumprod / np.concatenate([[1.0], alphas_cumprod[:-1]])
            betas = 1.0 - alphas
        
        return betas
    
    def convert_fm_t_to_dm_t(self, t):
        """
        Convert FM time t ∈ [0,1] to diffusion timestep t ∈ [0, 1000]
        Uses rectified flow alignment for trajectory matching
        """
        device = t.device
        rectified_alphas = self.rectified_alphas_cumprod_full.to(device)
        
        # Reverse for searchsorted (ascending order)
        rectified_alphas_rev = torch.flip(rectified_alphas, [0])
        
        # Ensure dtype and range
        t = t.to(rectified_alphas_rev.dtype)
        eps = torch.finfo(rectified_alphas_rev.dtype).eps
        t_clamped = t.clamp(min=rectified_alphas_rev[0] + eps, max=rectified_alphas_rev[-1] - eps)
        
        # Find corresponding diffusion timestep
        right_idx = torch.searchsorted(rectified_alphas_rev, t_clamped, right=True)
        # Clamp indices to valid range
        right_idx = right_idx.clamp(1, rectified_alphas_rev.shape[0] - 1)
        left_idx = right_idx - 1
        
        # Interpolate
        right_val = rectified_alphas_rev.gather(0, right_idx)
        left_val = rectified_alphas_rev.gather(0, left_idx)
        dm_t = left_idx + (t_clamped - left_val) / (right_val - left_val + 1e-8)
        
        # Reverse back to get actual diffusion timestep
        dm_t = self.num_timesteps - dm_t
        return dm_t.clamp(0, self.num_timesteps - 1)
    
    def convert_fm_xt_to_dm_xt(self, fm_xt, fm_t):
        """
        Convert FM trajectory point to diffusion trajectory point
        Applies scaling to align trajectory spaces
        """
        device = fm_xt.device
        scale = (self.sqrt_alphas_cumprod_full + self.sqrt_one_minus_alphas_cumprod_full).to(device)
        
        dm_t = self.convert_fm_t_to_dm_t(fm_t)
        
        # Linear interpolation for continuous timesteps
        dm_t_left = torch.floor(dm_t).long()
        dm_t_right = torch.ceil(dm_t).long()
        
        scale_left = scale[dm_t_left].view(-1, 1, 1, 1)
        scale_right = scale[dm_t_right].view(-1, 1, 1, 1)
        
        # Interpolate scale
        alpha = (dm_t - dm_t_left.float()).view(-1, 1, 1, 1)
        scale_t = scale_left + alpha * (scale_right - scale_left)
        
        # Apply scaling
        dm_xt = fm_xt * scale_t
        # Ensure dtype consistency with inputs (match UNet expected dtype)
        return dm_xt.to(dtype=fm_xt.dtype)
    
    def predict_start_from_v(self, x_t, t, v):
        """Predict x_0 from v-parameterization"""
        device = x_t.device
        sqrt_alphas = self.sqrt_alphas_cumprod.to(device)
        sqrt_one_minus_alphas = self.sqrt_one_minus_alphas_cumprod.to(device)
        
        return (
            sqrt_alphas[t.long()].view(-1, 1, 1, 1) * x_t -
            sqrt_one_minus_alphas[t.long()].view(-1, 1, 1, 1) * v
        )
    
    def predict_eps_from_v(self, x_t, t, v):
        """Predict noise from v-parameterization"""
        device = x_t.device
        sqrt_alphas = self.sqrt_alphas_cumprod.to(device)
        sqrt_one_minus_alphas = self.sqrt_one_minus_alphas_cumprod.to(device)
        
        return (
            sqrt_alphas[t.long()].view(-1, 1, 1, 1) * v +
            sqrt_one_minus_alphas[t.long()].view(-1, 1, 1, 1) * x_t
        )
    
    def predict_start_from_eps(self, x_t, t, eps):
        """Predict x_0 from eps-parameterization"""
        device = x_t.device
        sqrt_recip_alphas = self.sqrt_recip_alphas_cumprod.to(device)
        sqrt_recipm1_alphas = self.sqrt_recipm1_alphas_cumprod.to(device)
        
        return (
            sqrt_recip_alphas[t.long()].view(-1, 1, 1, 1) * x_t -
            sqrt_recipm1_alphas[t.long()].view(-1, 1, 1, 1) * eps
        )
    
    def get_vector_field_from_diffusion_pred(self, diffusion_pred, dm_xt, dm_t):
        """
        Convert diffusion model prediction to FM velocity field.
        This is the core Diff2Flow transformation: v_FM = x_0 - ε
        """
        if self.parameterization == "v":
            # v-parameterization: v = √ᾱ * ε - √(1-ᾱ) * x_0
            x_0_pred = self.predict_start_from_v(dm_xt, dm_t, diffusion_pred)
            eps_pred = self.predict_eps_from_v(dm_xt, dm_t, diffusion_pred)
        elif self.parameterization == "eps":
            # eps-parameterization
            x_0_pred = self.predict_start_from_eps(dm_xt, dm_t, diffusion_pred)
            eps_pred = diffusion_pred
        else:
            raise ValueError(f"Unknown parameterization: {self.parameterization}")
        
        # FM velocity field: v = x_0 - ε (rectified flow target)
        vector_field = x_0_pred - eps_pred
        return vector_field
    
    def sample_vector_field(self, fm_xt, fm_t, text_embedding, vector_embedding):
        """
        Sample vector field at FM trajectory point (fm_xt, fm_t)
        by converting to diffusion space and back
        """
        # Convert FM trajectory to diffusion trajectory
        dm_t = self.convert_fm_t_to_dm_t(fm_t)
        dm_xt = self.convert_fm_xt_to_dm_xt(fm_xt, fm_t)
        
        # Get diffusion model prediction
        # Assumes UNet and embeddings are already on the correct device; avoid per-step device transfers
        if dm_xt.dtype != vector_embedding.dtype:
            dm_xt = dm_xt.to(dtype=vector_embedding.dtype)
        dm_t_long = dm_t.long()

        diffusion_pred = self.unet(dm_xt, dm_t_long, text_embedding, vector_embedding)
        
        # Handle NaN values
        if torch.isnan(diffusion_pred).any():
            logger.warning("NaN detected in diffusion prediction, replacing with zeros")
            diffusion_pred = torch.nan_to_num(diffusion_pred, 0.0)
        
        # Convert to FM velocity field
        vector_field = self.get_vector_field_from_diffusion_pred(diffusion_pred, dm_xt, dm_t)
        
        return vector_field


# ---------------------------------

def train(args):
    train_util.verify_training_args(args)
    train_util.prepare_dataset_args(args, True)
    sdxl_train_util.verify_sdxl_training_args(args)
    deepspeed_utils.prepare_deepspeed_args(args)
    setup_logging(args, reset=True)

    assert (
        not args.weighted_captions or not args.cache_text_encoder_outputs
    ), "weighted_captions is not supported when caching text encoder outputs / cache_text_encoder_outputsを使うときはweighted_captionsはサポートされていません"
    assert (
        not args.train_text_encoder or not args.cache_text_encoder_outputs
    ), "cache_text_encoder_outputs is not supported when training text encoder / text encoderを学習するときはcache_text_encoder_outputsはサポートされていません"

    if args.block_lr:
        block_lrs = [float(lr) for lr in args.block_lr.split(",")]
        assert (
            len(block_lrs) == UNET_NUM_BLOCKS_FOR_BLOCK_LR
        ), f"block_lr must have {UNET_NUM_BLOCKS_FOR_BLOCK_LR} values / block_lrは{UNET_NUM_BLOCKS_FOR_BLOCK_LR}個の値を指定してください"
    else:
        block_lrs = None

    cache_latents = args.cache_latents
    use_dreambooth_method = args.in_json is None

    if args.seed is not None:
        set_seed(args.seed)

    tokenize_strategy = strategy_sdxl.SdxlTokenizeStrategy(args.max_token_length, args.tokenizer_cache_dir)
    strategy_base.TokenizeStrategy.set_strategy(tokenize_strategy)
    tokenizers = [tokenize_strategy.tokenizer1, tokenize_strategy.tokenizer2]

    # prepare caching strategy
    if args.cache_latents:
        latents_caching_strategy = strategy_sd.SdSdxlLatentsCachingStrategy(
            False, args.cache_latents_to_disk, args.vae_batch_size, args.skip_cache_check
        )
        strategy_base.LatentsCachingStrategy.set_strategy(latents_caching_strategy)

    # dataset
    if args.dataset_class is None:
        blueprint_generator = BlueprintGenerator(ConfigSanitizer(True, True, args.masked_loss, True))
        if args.dataset_config is not None:
            logger.info(f"Load dataset config from {args.dataset_config}")
            user_config = config_util.load_user_config(args.dataset_config)
            ignored = ["train_data_dir", "in_json"]
            if any(getattr(args, attr) is not None for attr in ignored):
                logger.warning(
                    "ignore following options because config file is found: {0} / 設定ファイルが利用されるため以下のオプションは無視されます: {0}".format(
                        ", ".join(ignored)
                    )
                )
        else:
            if use_dreambooth_method:
                logger.info("Using DreamBooth method.")
                user_config = {
                    "datasets": [
                        {
                            "subsets": config_util.generate_dreambooth_subsets_config_by_subdirs(
                                args.train_data_dir, args.reg_data_dir
                            )
                        }
                    ]
                }
            else:
                logger.info("Training with captions.")
                user_config = {
                    "datasets": [
                        {
                            "subsets": [
                                {
                                    "image_dir": args.train_data_dir,
                                    "metadata_file": args.in_json,
                                }
                            ]
                        }
                    ]
                }

        blueprint = blueprint_generator.generate(user_config, args)
        train_dataset_group, val_dataset_group = config_util.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    else:
        train_dataset_group = train_util.load_arbitrary_dataset(args)
        val_dataset_group = None

    current_epoch = Value("i", 0)
    current_step = Value("i", 0)
    ds_for_collator = train_dataset_group if args.max_data_loader_n_workers == 0 else None
    collator = train_util.collator_class(current_epoch, current_step, ds_for_collator)

    train_dataset_group.verify_bucket_reso_steps(32)

    if args.debug_dataset:
        train_util.debug_dataset(train_dataset_group, True)
        return
    if len(train_dataset_group) == 0:
        logger.error(
            "No data found. Please verify the metadata file and train_data_dir option. / 画像がありません。メタデータおよびtrain_data_dirオプションを確認してください。"
        )
        return

    if cache_latents:
        assert (
            train_dataset_group.is_latent_cacheable()
        ), "when caching latents, either color_aug or random_crop cannot be used / latentをキャッシュするときはcolor_augとrandom_cropは使えません"

    if args.cache_text_encoder_outputs:
        assert (
            train_dataset_group.is_text_encoder_output_cacheable()
        ), "when caching text encoder output, either caption_dropout_rate, shuffle_caption, token_warmup_step or caption_tag_dropout_rate cannot be used / text encoderの出力をキャッシュするときはcaption_dropout_rate, shuffle_caption, token_warmup_step, caption_tag_dropout_rateは使えません"

    # accelerator
    logger.info("prepare accelerator")
    accelerator = train_util.prepare_accelerator(args)

    # dtypes
    weight_dtype, save_dtype = train_util.prepare_dtype(args)
    vae_dtype = torch.float32 if args.no_half_vae else weight_dtype

    # load models
    (
        load_stable_diffusion_format,
        text_encoder1,
        text_encoder2,
        vae,
        unet,
        logit_scale,
        ckpt_info,
    ) = sdxl_train_util.load_target_model(args, accelerator, "sdxl", weight_dtype)

    # Ensure UNet is on the accelerator device early (before wrappers/sampling)
    # dtype is handled below depending on whether we train UNet
    unet.to(accelerator.device)

    # verify load/save model formats
    if load_stable_diffusion_format:
        src_stable_diffusion_ckpt = args.pretrained_model_name_or_path
        src_diffusers_model_path = None
    else:
        src_stable_diffusion_ckpt = None
        src_diffusers_model_path = args.pretrained_model_name_or_path

    if args.save_model_as is None:
        save_stable_diffusion_format = load_stable_diffusion_format
        use_safetensors = args.use_safetensors
    else:
        save_stable_diffusion_format = args.save_model_as.lower() == "ckpt" or args.save_model_as.lower() == "safetensors"
        use_safetensors = args.use_safetensors or ("safetensors" in args.save_model_as.lower())

    # memory efficient attention etc.
    if args.diffusers_xformers:
        accelerator.print("Use xformers by Diffusers")
        # set_diffusers_xformers_flag(unet, True)  # unet is original
        def _set_flag(model, valid):
            def fn_recursive_set_mem_eff(module: torch.nn.Module):
                if hasattr(module, "set_use_memory_efficient_attention_xformers"):
                    module.set_use_memory_efficient_attention_xformers(valid)
                for child in module.children():
                    fn_recursive_set_mem_eff(child)
            fn_recursive_set_mem_eff(model)
        _set_flag(vae, True)
    else:
        accelerator.print("Disable Diffusers' xformers")
        train_util.replace_unet_modules(unet, args.mem_eff_attn, args.xformers, args.sdpa)
        if torch.__version__ >= "2.0.0":
            vae.set_use_memory_efficient_attention_xformers(args.xformers)

    # Replacement above may instantiate new CPU modules inside UNet; ensure it's on the correct device
    unet.to(accelerator.device)

    # cache latents
    if cache_latents:
        vae.to(accelerator.device, dtype=vae_dtype)
        vae.requires_grad_(False)
        vae.eval()

        train_dataset_group.new_cache_latents(vae, accelerator)

        vae.to("cpu")
        clean_memory_on_device(accelerator.device)
        accelerator.wait_for_everyone()

    # models train/eval prep
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    train_unet = args.learning_rate != 0
    train_text_encoder1 = False
    train_text_encoder2 = False

    text_encoding_strategy = strategy_sdxl.SdxlTextEncodingStrategy()
    strategy_base.TextEncodingStrategy.set_strategy(text_encoding_strategy)

    if args.train_text_encoder:
        accelerator.print("enable text encoder training")
        if args.gradient_checkpointing:
            text_encoder1.gradient_checkpointing_enable()
            text_encoder2.gradient_checkpointing_enable()
        lr_te1 = args.learning_rate_te1 if args.learning_rate_te1 is not None else args.learning_rate
        lr_te2 = args.learning_rate_te2 if args.learning_rate_te2 is not None else args.learning_rate
        train_text_encoder1 = lr_te1 != 0
        train_text_encoder2 = lr_te2 != 0

        if not train_text_encoder1:
            text_encoder1.to(weight_dtype)
        if not train_text_encoder2:
            text_encoder2.to(weight_dtype)
        text_encoder1.requires_grad_(train_text_encoder1)
        text_encoder2.requires_grad_(train_text_encoder2)
        text_encoder1.train(train_text_encoder1)
        text_encoder2.train(train_text_encoder2)
    else:
        text_encoder1.to(weight_dtype)
        text_encoder2.to(weight_dtype)
        text_encoder1.requires_grad_(False)
        text_encoder2.requires_grad_(False)
        text_encoder1.eval()
        text_encoder2.eval()

        # cache text encoder outputs
        if args.cache_text_encoder_outputs:
            text_encoder_output_caching_strategy = strategy_sdxl.SdxlTextEncoderOutputsCachingStrategy(
                args.cache_text_encoder_outputs_to_disk, None, False, is_weighted=args.weighted_captions
            )
            strategy_base.TextEncoderOutputsCachingStrategy.set_strategy(text_encoder_output_caching_strategy)

            text_encoder1.to(accelerator.device)
            text_encoder2.to(accelerator.device)
            with accelerator.autocast():
                train_dataset_group.new_cache_text_encoder_outputs([text_encoder1, text_encoder2], accelerator)

        accelerator.wait_for_everyone()

    if not cache_latents:
        vae.requires_grad_(False)
        vae.eval()
        vae.to(accelerator.device, dtype=vae_dtype)

    unet.requires_grad_(train_unet)
    if not train_unet:
        unet.to(accelerator.device, dtype=weight_dtype)

    training_models = []
    params_to_optimize = []
    if train_unet:
        training_models.append(unet)
        if block_lrs is None:
            params_to_optimize.append({"params": list(unet.parameters()), "lr": args.learning_rate})
        else:
            params_to_optimize.extend(get_block_params_to_optimize(unet, block_lrs))

    if train_text_encoder1:
        training_models.append(text_encoder1)
        params_to_optimize.append({"params": list(text_encoder1.parameters()), "lr": args.learning_rate_te1 or args.learning_rate})
    if train_text_encoder2:
        training_models.append(text_encoder2)
        params_to_optimize.append({"params": list(text_encoder2.parameters()), "lr": args.learning_rate_te2 or args.learning_rate})

    # stats
    n_params = 0
    for group in params_to_optimize:
        for p in group["params"]:
            n_params += p.numel()

    accelerator.print(f"train unet: {train_unet}, text_encoder1: {train_text_encoder1}, text_encoder2: {train_text_encoder2}")
    accelerator.print(f"number of models: {len(training_models)}")
    accelerator.print(f"number of trainable parameters: {n_params}")

    # optimizer, dataloader etc.
    accelerator.print("prepare optimizer, data loader etc.")

    if args.fused_optimizer_groups:
        # fused backward pass setup
        n_total_params = sum(len(params["params"]) for params in params_to_optimize)
        params_per_group = math.ceil(n_total_params / args.fused_optimizer_groups)

        grouped_params = []
        param_group = []
        param_group_lr = -1
        for group in params_to_optimize:
            lr = group["lr"]
            for p in group["params"]:
                if lr != param_group_lr:
                    if param_group:
                        grouped_params.append({"params": param_group, "lr": param_group_lr})
                        param_group = []
                    param_group_lr = lr
                param_group.append(p)
                if len(param_group) >= params_per_group:
                    grouped_params.append({"params": param_group, "lr": param_group_lr})
                    param_group = []
        if param_group:
            grouped_params.append({"params": param_group, "lr": param_group_lr})
        params_to_optimize = grouped_params

    optimizer_name, optimizer_args, optimizer = train_util.get_optimizer(args, params_to_optimize)

    if args.fused_optimizer_groups:
        # create multiple optimizers with the same hyperparameters for fused backward pass
        optimizers = [train_util.get_optimizer(args, [{"params": group["params"], "lr": group["lr"]}])[2] for group in params_to_optimize]
        # use the first optimizer/scheduler in the common code path
        optimizer = optimizers[0]
        # counts are computed later when registering hooks

    # prepare dataloader
    # strategies are set here because they cannot be referenced in another process. Copy them with the dataset
    # some strategies can be None
    train_dataset_group.set_current_strategies()

    # DataLoaderのプロセス数：0 は persistent_workers が使えないので注意
    n_workers = min(args.max_data_loader_n_workers, os.cpu_count())  # cpu_count or max_data_loader_n_workers
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset_group,
        batch_size=1,
        shuffle=True,
        collate_fn=collator,
        num_workers=n_workers,
        persistent_workers=args.persistent_data_loader_workers,
    )

    # 学習ステップ数を計算する
    if args.max_train_epochs is not None:
        args.max_train_steps = args.max_train_epochs * math.ceil(
            len(train_dataloader) / accelerator.num_processes / args.gradient_accumulation_steps
        )
        accelerator.print(
            f"override steps. steps for {args.max_train_epochs} epochs is / 指定エポックまでのステップ数: {args.max_train_steps}"
        )

    # データセット側にも学習ステップを送信
    train_dataset_group.set_max_train_steps(args.max_train_steps)

    # lr schedulerを用意する
    if args.fused_optimizer_groups:
        # prepare lr schedulers for each optimizer
        lr_schedulers = [train_util.get_scheduler_fix(args, opt, accelerator.num_processes) for opt in optimizers]
        lr_scheduler = lr_schedulers[0]  # avoid error in the following code
    else:
        lr_scheduler = train_util.get_scheduler_fix(args, optimizer, accelerator.num_processes)

    # Prepare models/optimizer/dataloader/scheduler with accelerator (and deepspeed)
    if args.deepspeed:
        ds_model = deepspeed_utils.prepare_deepspeed_model(
            args,
            unet=unet if train_unet else None,
            text_encoder1=text_encoder1 if train_text_encoder1 else None,
            text_encoder2=text_encoder2 if train_text_encoder2 else None,
        )
        ds_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            ds_model, optimizer, train_dataloader, lr_scheduler
        )
        training_models = [ds_model]
        # Ensure fused group index 0 references the prepared optimizer/scheduler (deepspeed path)
        if args.fused_optimizer_groups:
            optimizers[0] = optimizer
            lr_schedulers[0] = lr_scheduler
    else:
        if train_unet:
            unet = accelerator.prepare(unet)
        if train_text_encoder1:
            text_encoder1 = accelerator.prepare(text_encoder1)
        if train_text_encoder2:
            text_encoder2 = accelerator.prepare(text_encoder2)
        optimizer, train_dataloader, lr_scheduler = accelerator.prepare(optimizer, train_dataloader, lr_scheduler)

        # rebuild training_models with prepared modules
        training_models = []
        if train_unet:
            training_models.append(unet)
        if train_text_encoder1:
            training_models.append(text_encoder1)
        if train_text_encoder2:
            training_models.append(text_encoder2)

        # Ensure fused group index 0 references the prepared optimizer/scheduler
        if args.fused_optimizer_groups:
            optimizers[0] = optimizer
            lr_schedulers[0] = lr_scheduler

    if args.fused_optimizer_groups and not args.fused_backward_pass:
        # prepare additional optimizers and lr schedulers for fused backward pass
        for i in range(1, len(optimizers)):
            optimizers[i] = accelerator.prepare(optimizers[i])
            lr_schedulers[i] = accelerator.prepare(lr_schedulers[i])

        # counters and hooks are used to determine when to step each optimizer
        optimizer_hooked_count = {}
        optimizer_stepped = {}
        num_parameters_per_group = [0] * len(optimizers)
        parameter_optimizer_map = {}

        for opt_idx, opt in enumerate(optimizers):
            for param_group in opt.param_groups:
                for parameter in param_group["params"]:
                    if parameter.requires_grad:

                        def optimizer_hook(parameter: torch.Tensor):
                            if accelerator.sync_gradients and args.max_grad_norm != 0.0:
                                accelerator.clip_grad_norm_(parameter, args.max_grad_norm)

                            # Only count/step on the synchronized accumulation step
                            if not accelerator.sync_gradients:
                                return

                            i = parameter_optimizer_map[parameter]
                            optimizer_hooked_count[i] += 1
                            if optimizer_hooked_count[i] == num_parameters_per_group[i]:
                                if getattr(args, "debug_devices", False):
                                    accelerator.print(f"[debug] fused opt[{i}] stepping (params={num_parameters_per_group[i]})")
                                _t0 = time.perf_counter() if getattr(args, "debug_devices", False) else None
                                if getattr(args, "debug_devices", False):
                                    # dump traceback if step stalls
                                    faulthandler.enable()
                                    faulthandler.dump_traceback_later(60, file=sys.stderr)
                                try:
                                    optimizers[i].step()
                                finally:
                                    if getattr(args, "debug_devices", False):
                                        faulthandler.cancel_dump_traceback_later()
                                if getattr(args, "debug_devices", False) and _t0 is not None:
                                    accelerator.print(f"[debug] fused opt[{i}] step done in {time.perf_counter()-_t0:.3f}s")
                                optimizers[i].zero_grad(set_to_none=True)
                                optimizer_stepped[i] = True

                        parameter.register_post_accumulate_grad_hook(optimizer_hook)
                        parameter_optimizer_map[parameter] = opt_idx
                        num_parameters_per_group[opt_idx] += 1

    if getattr(args, "debug_devices", False) and accelerator.is_local_main_process:
        try:
            accelerator.print(
                f"[debug] fused optimizer groups setup: {len(optimizers)} groups; expected per-group params: {num_parameters_per_group}"
            )
        except Exception:
            pass

    # resume
    train_util.resume_from_local_or_hf_if_specified(accelerator, args)

    if args.fused_backward_pass:
        # use fused optimizer for backward pass: other optimizers will be supported in the future
        import library.adafactor_fused

        library.adafactor_fused.patch_adafactor_fused(optimizer)
        for param_group in optimizer.param_groups:
            for parameter in param_group["params"]:
                if parameter.requires_grad:

                    def __grad_hook(tensor: torch.Tensor, param_group=param_group):
                        if accelerator.sync_gradients and args.max_grad_norm != 0.0:
                            accelerator.clip_grad_norm_(tensor, args.max_grad_norm)
                        optimizer.step_param(tensor, param_group)
                        tensor.grad = None

                    parameter.register_post_accumulate_grad_hook(__grad_hook)

    # epochs calculation
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    if (args.save_n_epoch_ratio is not None) and (args.save_n_epoch_ratio > 0):
        args.save_every_n_epochs = math.floor(num_train_epochs / args.save_n_epoch_ratio) or 1

    # training header
    accelerator.print("running training / 学習開始")
    accelerator.print(f"  num examples / サンプル数: {train_dataset_group.num_train_images}")
    accelerator.print(f"  num batches per epoch / 1epochのバッチ数: {len(train_dataloader)}")
    accelerator.print(f"  num epochs / epoch数: {num_train_epochs}")
    accelerator.print(
        f"  batch size per device / バッチサイズ: {', '.join([str(d.batch_size) for d in train_dataset_group.datasets])}"
    )
    accelerator.print(f"  gradient accumulation steps / 勾配を合計するステップ数 = {args.gradient_accumulation_steps}")
    accelerator.print(f"  total optimization steps / 学習ステップ数: {args.max_train_steps}")

    progress_bar = tqdm(range(args.max_train_steps), smoothing=0, disable=not accelerator.is_local_main_process, desc="steps")
    global_step = 0

    # Optional profiler setup
    prof = None
    profiler_dir = getattr(args, "profile_dir", "profiler_logs")
    if getattr(args, "profile", False) and accelerator.is_local_main_process:
        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)
        os.makedirs(profiler_dir, exist_ok=True)
        prof = profile(
            activities=activities,
            schedule=schedule(
                wait=getattr(args, "profile_wait", 1),
                warmup=getattr(args, "profile_warmup", 1),
                active=getattr(args, "profile_active", 5),
                repeat=1,
            ),
            on_trace_ready=tensorboard_trace_handler(getattr(args, "profile_dir", "profiler_logs")),
            record_shapes=True,
            profile_memory=True,
        )
        prof.start()
        accelerator.print(f"Starting profiler at {profiler_dir}")

    # Initialize FlowMatch scheduler (wrapper is created after accelerator.prepare)
    noise_scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=args.fm_shift)
    
    # Create Diff2Flow wrapper now that models are prepared and scheduler exists
    if getattr(args, 'use_diff2flow', True):  # Default to True for better results
        accelerator.print("Using Diff2Flow for knowledge transfer from pre-trained diffusion model")
        diff2flow_wrapper = Diff2FlowWrapper(
            unet=unet,
            noise_scheduler=noise_scheduler,
            parameterization=getattr(args, 'diffusion_parameterization', 'v')
        )
    else:
        diff2flow_wrapper = None
        accelerator.print("Using naive Flow Matching (training from scratch)")

    # Optional: device/dtype debug after prepare
    if getattr(args, "debug_devices", False) and accelerator.is_local_main_process:
        try:
            def _param_device(model):
                try:
                    return next(model.parameters()).device
                except StopIteration:
                    return torch.device("cpu")

            unet_dev = _param_device(unet) if 'unet' in locals() and train_unet else None
            te1_dev = _param_device(text_encoder1) if 'text_encoder1' in locals() and train_text_encoder1 else None
            te2_dev = _param_device(text_encoder2) if 'text_encoder2' in locals() and train_text_encoder2 else None

            def _opt_devices(opt):
                devs = set()
                for g in opt.param_groups:
                    for p in g.get("params", []):
                        if isinstance(p, torch.Tensor):
                            devs.add(str(p.device))
                return sorted(devs)

            opt0_devs = _opt_devices(optimizer)
            accelerator.print(f"[debug] accelerator.device={accelerator.device} | unet={unet_dev} te1={te1_dev} te2={te2_dev} | opt0 param devices={opt0_devs}")
            if args.fused_optimizer_groups:
                for i, opt in enumerate(optimizers):
                    accelerator.print(f"[debug] fused opt[{i}] param devices={_opt_devices(opt)}")
        except Exception as e:
            accelerator.print(f"[debug] device dump failed: {e}")

    if accelerator.is_main_process:
        init_kwargs = {}
        if args.wandb_run_name:
            init_kwargs["wandb"] = {"name": args.wandb_run_name}
        if args.log_tracker_config is not None:
            init_kwargs = toml.load(args.log_tracker_config)
        accelerator.init_trackers(
            "finetuning_fm" if args.log_tracker_name is None else args.log_tracker_name,
            config=train_util.get_sanitized_config_or_none(args),
            init_kwargs=init_kwargs,
        )

    # For --sample_at_first
    sdxl_train_util.sample_images(
        accelerator, args, 0, global_step, accelerator.device, vae, tokenizers, [text_encoder1, text_encoder2], unet
    )
    if len(accelerator.trackers) > 0:
        accelerator.log({}, step=0)

    # training loop
    loss_recorder = train_util.LossRecorder()
    for epoch in range(num_train_epochs):
        accelerator.print(f"\nepoch {epoch+1}/{num_train_epochs}")
        current_epoch.value = epoch + 1

        for m in training_models:
            m.train()

        for step, batch in enumerate(train_dataloader):
            current_step.value = global_step

            if args.fused_optimizer_groups:
                optimizer_hooked_count = {i: 0 for i in range(len(optimizers))}
                optimizer_stepped = {i: False for i in range(len(optimizers))}

            with accelerator.accumulate(*training_models):
                # latents
                if "latents" in batch and batch["latents"] is not None:
                    latents = batch["latents"].to(accelerator.device).to(dtype=weight_dtype)
                else:
                    with torch.no_grad():
                        latents = vae.encode(batch["images"].to(vae_dtype)).latent_dist.sample().to(weight_dtype)
                        if torch.any(torch.isnan(latents)):
                            accelerator.print("NaN found in latents, replacing with zeros")
                            latents = torch.nan_to_num(latents, 0, out=latents)
                latents = latents * sdxl_model_util.VAE_SCALE_FACTOR

                # text cond
                text_encoder_outputs_list = batch.get("text_encoder_outputs_list", None)
                if text_encoder_outputs_list is not None:
                    encoder_hidden_states1, encoder_hidden_states2, pool2 = text_encoder_outputs_list
                    encoder_hidden_states1 = encoder_hidden_states1.to(accelerator.device, dtype=weight_dtype)
                    encoder_hidden_states2 = encoder_hidden_states2.to(accelerator.device, dtype=weight_dtype)
                    pool2 = pool2.to(accelerator.device, dtype=weight_dtype)
                else:
                    input_ids1, input_ids2 = batch["input_ids_list"]
                    with torch.set_grad_enabled(args.train_text_encoder):
                        if args.weighted_captions:
                            input_ids_list, weights_list = tokenize_strategy.tokenize_with_weights(batch["captions"])
                            encoder_hidden_states1, encoder_hidden_states2, pool2 = (
                                text_encoding_strategy.encode_tokens_with_weights(
                                    tokenize_strategy,
                                    [text_encoder1, text_encoder2, accelerator.unwrap_model(text_encoder2)],
                                    input_ids_list,
                                    weights_list,
                                )
                            )
                        else:
                            input_ids1 = input_ids1.to(accelerator.device)
                            input_ids2 = input_ids2.to(accelerator.device)
                            encoder_hidden_states1, encoder_hidden_states2, pool2 = text_encoding_strategy.encode_tokens(
                                tokenize_strategy,
                                [text_encoder1, text_encoder2, accelerator.unwrap_model(text_encoder2)],
                                [input_ids1, input_ids2],
                            )
                        # Ensure tensors are on the accelerator device and correct dtype prior to concat/calls
                        encoder_hidden_states1 = encoder_hidden_states1.to(device=accelerator.device, dtype=weight_dtype)
                        encoder_hidden_states2 = encoder_hidden_states2.to(device=accelerator.device, dtype=weight_dtype)
                        pool2 = pool2.to(device=accelerator.device, dtype=weight_dtype)

                # size embeddings
                orig_size = batch["original_sizes_hw"]
                crop_size = batch["crop_top_lefts"]
                target_size = batch["target_sizes_hw"]
                embs = sdxl_train_util.get_size_embeddings(
                    orig_size, crop_size, target_size, accelerator.device
                ).to(device=accelerator.device, dtype=weight_dtype)

                # concat embeddings
                vector_embedding = torch.cat([pool2, embs], dim=1).to(device=accelerator.device, dtype=weight_dtype)
                text_embedding = torch.cat([encoder_hidden_states1, encoder_hidden_states2], dim=2).to(
                    device=accelerator.device, dtype=weight_dtype
                )

                bsz = latents.shape[0]
                
                if diff2flow_wrapper is not None:
                    # Diff2Flow training approach
                    # Sample FM time uniformly
                    fm_t = torch.rand(bsz, device=latents.device, dtype=latents.dtype)
                    
                    # Create FM interpolation: x_t = (1-t)*x_0 + t*x_1
                    noise = torch.randn_like(latents)
                    fm_xt = (1.0 - fm_t.view(-1, 1, 1, 1)) * noise + fm_t.view(-1, 1, 1, 1) * latents
                    
                    # Diff2Flow target velocity: v = x_1 - x_0 = latents - noise
                    target = latents - noise
                    
                    # Get vector field prediction from Diff2Flow wrapper
                    with accelerator.autocast():
                        if getattr(args, "profile", False):
                            with record_function("diff2flow_forward"):
                                noise_pred = diff2flow_wrapper.sample_vector_field(
                                    fm_xt, fm_t, text_embedding, vector_embedding
                                )
                        else:
                            noise_pred = diff2flow_wrapper.sample_vector_field(
                                fm_xt, fm_t, text_embedding, vector_embedding
                            )
                else:
                    # Naive Flow Matching approach (original implementation)
                    u = compute_density_for_timestep_sampling(
                        weighting_scheme="logit_normal",
                        batch_size=bsz,
                        logit_mean=args.fm_logit_mean,
                        logit_std=args.fm_logit_std,
                        mode_scale=args.fm_mode_scale,
                    )
                    indices = (u * noise_scheduler.config.num_train_timesteps).long().clamp(
                        min=0, max=noise_scheduler.config.num_train_timesteps - 1
                    )
                    timesteps = noise_scheduler.timesteps[indices].to(device=latents.device)
                    
                    # build z_t and target
                    sigmas = _get_sigmas(noise_scheduler, latents.device, timesteps, n_dim=latents.ndim, dtype=latents.dtype)
                    noise = torch.randn(latents.size(), device=latents.device, dtype=latents.dtype)
                    noisy_latents = (1.0 - sigmas) * latents + sigmas * noise
                    
                    noisy_latents = noisy_latents.to(weight_dtype)
                    
                    # forward UNet
                    with accelerator.autocast():
                        if getattr(args, "profile", False):
                            with record_function("unet_forward"):
                                noise_pred = unet(noisy_latents, timesteps, text_embedding, vector_embedding)
                        else:
                            noise_pred = unet(noisy_latents, timesteps, text_embedding, vector_embedding)
                    
                    # FM target (velocity)
                    target = noise - latents

                # Huber threshold (avoid 'snr' schedule with FM)
                huber_c = None
                if args.loss_type in ("huber", "smooth_l1"):
                    if getattr(args, "huber_schedule", "constant") == "snr":
                        logger.warning("Huber schedule 'snr' is not supported with Flow Matching; using constant.")
                        huber_c = torch.full((bsz,), args.huber_c * args.huber_scale, device=latents.device)
                    else:
                        huber_c = train_util.get_huber_threshold_if_needed(args, timesteps, noise_scheduler)

                # compute loss, skip SNR/vpred/debiased for FM
                if args.masked_loss or ("alpha_masks" in batch and batch.get("alpha_masks") is not None):
                    loss = train_util.conditional_loss(noise_pred.float(), target.float(), args.loss_type, "none", huber_c)
                    loss = apply_masked_loss(loss, batch)
                    loss = loss.mean([1, 2, 3]).mean()
                else:
                    loss = train_util.conditional_loss(noise_pred.float(), target.float(), args.loss_type, "mean", huber_c)

                global _warned_min_snr_gamma, _warned_vpred_flags
                if getattr(args, "min_snr_gamma", 0) and not _warned_min_snr_gamma:
                    logger.warning("min_snr_gamma is not applicable to Flow Matching; skipping SNR weighting.")
                    _warned_min_snr_gamma = True
                if (
                    getattr(args, "scale_v_pred_loss_like_noise_pred", False)
                    or getattr(args, "v_pred_like_loss", None)
                    or getattr(args, "debiased_estimation_loss", None)
                ) and not _warned_vpred_flags:
                    logger.warning("v-pred-like or debiased estimation flags are not applied in Flow Matching; skipping.")
                    _warned_vpred_flags = True

                accelerator.backward(loss)

                if not (args.fused_backward_pass or args.fused_optimizer_groups):
                    if accelerator.sync_gradients and args.max_grad_norm != 0.0:
                        params_to_clip = []
                        for m in training_models:
                            params_to_clip.extend(m.parameters())
                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                    if getattr(args, "debug_devices", False):
                        accelerator.print("[debug] before optimizer.step()")
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        _t0 = time.perf_counter()
                    if getattr(args, "debug_devices", False):
                        faulthandler.enable()
                        faulthandler.dump_traceback_later(60, file=sys.stderr)
                    try:
                        optimizer.step()
                    finally:
                        if getattr(args, "debug_devices", False):
                            faulthandler.cancel_dump_traceback_later()
                    if getattr(args, "debug_devices", False):
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        accelerator.print(f"[debug] optimizer.step() took {time.perf_counter()-_t0:.3f}s")
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                else:
                    lr_scheduler.step()
                    if args.fused_optimizer_groups:
                        for i in range(1, len(optimizers)):
                            lr_schedulers[i].step()

            # Fallback: if any fused optimizer group wasn't stepped by the hook, step it now
            if args.fused_optimizer_groups and accelerator.sync_gradients:
                for i in range(len(optimizers)):
                    if not optimizer_stepped.get(i, False):
                        if getattr(args, "debug_devices", False):
                            hooked = optimizer_hooked_count.get(i, None)
                            accelerator.print(
                                f"[debug] fused opt[{i}] fallback step (stepping now) hooked={hooked}/{num_parameters_per_group[i]}"
                            )
                        if torch.cuda.is_available() and not getattr(args, "fused_hook_no_cuda_sync", False):
                            torch.cuda.synchronize()
                        _t0 = time.perf_counter()
                        if getattr(args, "debug_devices", False):
                            faulthandler.enable()
                            faulthandler.dump_traceback_later(getattr(args, "fused_hook_timeout", 60), file=sys.stderr)
                        try:
                            optimizers[i].step()
                        finally:
                            if getattr(args, "debug_devices", False):
                                faulthandler.cancel_dump_traceback_later()
                        if torch.cuda.is_available() and not getattr(args, "fused_hook_no_cuda_sync", False):
                            torch.cuda.synchronize()
                        if getattr(args, "debug_devices", False):
                            accelerator.print(
                                f"[debug] fused opt[{i}] fallback step done in {time.perf_counter()-_t0:.3f}s"
                            )
                        optimizers[i].zero_grad(set_to_none=True)
                        optimizer_stepped[i] = True

            # post-step
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # profiler step
                if prof is not None:
                    prof.step()

                sdxl_train_util.sample_images(
                    accelerator,
                    args,
                    None,
                    global_step,
                    accelerator.device,
                    vae,
                    tokenizers,
                    [text_encoder1, text_encoder2],
                    unet,
                )

                # Debug fused optimizer groups stepping
                if args.fused_optimizer_groups and getattr(args, "debug_devices", False) and accelerator.is_local_main_process:
                    not_stepped = [i for i, s in optimizer_stepped.items() if not s]
                    if not_stepped:
                        accelerator.print(f"[debug] fused optimizer groups not stepped this round. hooked_counts={optimizer_hooked_count} expected_per_group={num_parameters_per_group} missing={not_stepped}")

                # if current step is end of training, dont save since it will be saved later
                if args.save_every_n_steps is not None and global_step % args.save_every_n_steps == 0 and global_step is not args.save_every_n_steps:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        src_path = src_stable_diffusion_ckpt if save_stable_diffusion_format else src_diffusers_model_path
                        sdxl_train_util.save_sd_model_on_epoch_end_or_stepwise(
                            args,
                            False,
                            accelerator,
                            src_path,
                            save_stable_diffusion_format,
                            use_safetensors,
                            save_dtype,
                            epoch,
                            num_train_epochs,
                            global_step,
                            accelerator.unwrap_model(text_encoder1),
                            accelerator.unwrap_model(text_encoder2),
                            accelerator.unwrap_model(unet),
                            vae,
                            logit_scale,
                            ckpt_info,
                        )

            current_loss = loss.detach().item()
            if len(accelerator.trackers) > 0:
                logs = {"loss": current_loss}
                if block_lrs is None:
                    train_util.append_lr_to_logs(logs, lr_scheduler, args.optimizer_type, including_unet=train_unet)
                else:
                    append_block_lr_to_logs(block_lrs, logs, lr_scheduler, args.optimizer_type)
                accelerator.log(logs, step=global_step)

            loss_recorder.add(epoch=epoch, step=step, loss=current_loss)
            avr_loss: float = loss_recorder.moving_average
            logs = {"avr_loss": avr_loss}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        if len(accelerator.trackers) > 0:
            logs = {"loss/epoch": loss_recorder.moving_average}
            accelerator.log(logs, step=epoch + 1)

        accelerator.wait_for_everyone()

        if args.save_every_n_epochs is not None:
            if accelerator.is_main_process:
                src_path = src_stable_diffusion_ckpt if save_stable_diffusion_format else src_diffusers_model_path
                sdxl_train_util.save_sd_model_on_epoch_end_or_stepwise(
                    args,
                    True,
                    accelerator,
                    src_path,
                    save_stable_diffusion_format,
                    use_safetensors,
                    save_dtype,
                    epoch,
                    num_train_epochs,
                    global_step,
                    accelerator.unwrap_model(text_encoder1),
                    accelerator.unwrap_model(text_encoder2),
                    accelerator.unwrap_model(unet),
                    vae,
                    logit_scale,
                    ckpt_info,
                )

        sdxl_train_util.sample_images(
            accelerator,
            args,
            epoch + 1,
            global_step,
            accelerator.device,
            vae,
            tokenizers,
            [text_encoder1, text_encoder2],
            unet,
        )

    # Stop profiler if enabled
    if prof is not None:
        prof.stop()
        logger.info(f"Profiler trace(s) written to {args.profile_dir}")

    is_main_process = accelerator.is_main_process
    unet = accelerator.unwrap_model(unet)
    text_encoder1 = accelerator.unwrap_model(text_encoder1)
    text_encoder2 = accelerator.unwrap_model(text_encoder2)

    accelerator.end_training()

    if args.save_state or args.save_state_on_train_end:
        train_util.save_state_on_train_end(args, accelerator)

    del accelerator

    if is_main_process:
        src_path = src_stable_diffusion_ckpt if save_stable_diffusion_format else src_diffusers_model_path
        sdxl_train_util.save_sd_model_on_train_end(
            args,
            src_path,
            save_stable_diffusion_format,
            use_safetensors,
            save_dtype,
            epoch,
            global_step,
            text_encoder1,
            text_encoder2,
            unet,
            vae,
            logit_scale,
            ckpt_info,
        )
        logger.info("model saved.")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    add_logging_arguments(parser)
    train_util.add_sd_models_arguments(parser)
    train_util.add_dataset_arguments(parser, True, True, True)
    train_util.add_training_arguments(parser, False)
    train_util.add_masked_loss_arguments(parser)
    deepspeed_utils.add_deepspeed_arguments(parser)
    train_util.add_sd_saving_arguments(parser)
    train_util.add_optimizer_arguments(parser)
    config_util.add_config_arguments(parser)
    custom_train_functions.add_custom_train_arguments(parser)  # keep available if needed
    sdxl_train_util.add_sdxl_training_arguments(parser)

    parser.add_argument(
        "--learning_rate_te1",
        type=float,
        default=None,
        help="learning rate for text encoder 1 (ViT-L) / text encoder 1 (ViT-L)の学習率",
    )
    parser.add_argument(
        "--learning_rate_te2",
        type=float,
        default=None,
        help="learning rate for text encoder 2 (BiG-G) / text encoder 2 (BiG-G)の学習率",
    )

    parser.add_argument(
        "--diffusers_xformers", action="store_true", help="use xformers by diffusers / Diffusersでxformersを使用する"
    )
    parser.add_argument("--train_text_encoder", action="store_true", help="train text encoder / text encoderも学習する")
    parser.add_argument(
        "--no_half_vae",
        action="store_true",
        help="do not use fp16/bf16 VAE in mixed precision (use float VAE) / mixed precisionでも fp16/bf16 VAEを使わずfloat VAEを使う",
    )
    parser.add_argument(
        "--block_lr",
        type=str,
        default=None,
        help=f"learning rates for each block of U-Net, comma-separated, {UNET_NUM_BLOCKS_FOR_BLOCK_LR} values / "
        + f"U-Netの各ブロックの学習率、カンマ区切り、{UNET_NUM_BLOCKS_FOR_BLOCK_LR}個の値",
    )
    parser.add_argument(
        "--fused_optimizer_groups",
        type=int,
        default=None,
        help="number of optimizers for fused backward pass and optimizer step / fused backward passとoptimizer stepのためのoptimizer数",
    )

    # Flow Matching specific args
    parser.add_argument(
        "--fm_shift",
        type=float,
        default=3.0,
        help="FlowMatch scheduler shift value (recommended ~3.0)",
    )
    parser.add_argument(
        "--fm_logit_mean",
        type=float,
        default=0.0,
        help="logit-normal mean for timestep sampling",
    )
    parser.add_argument(
        "--fm_logit_std",
        type=float,
        default=1.0,
        help="logit-normal std for timestep sampling",
    )
    parser.add_argument(
        "--fm_mode_scale",
        type=float,
        default=1.29,
        help="mode scale used in logit-normal density",
    )
    
    # Diff2Flow specific args
    parser.add_argument(
        "--use_diff2flow",
        action="store_true",
        default=True,
        help="Use Diff2Flow for knowledge transfer from pre-trained diffusion model (default: True)",
    )
    parser.add_argument(
        "--naive_fm",
        action="store_true",
        help="Use naive Flow Matching instead of Diff2Flow (trains from scratch)",
    )
    parser.add_argument(
        "--diffusion_parameterization",
        type=str,
        choices=["v", "eps"],
        default="v",
        help="Diffusion model parameterization for Diff2Flow (v or eps)",
    )

    # Profiler options
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable torch.profiler (TensorBoard trace in --profile_dir)",
    )
    parser.add_argument(
        "--profile_dir",
        type=str,
        default="profiler_logs",
        help="Directory to save profiler traces (TensorBoard format)",
    )
    parser.add_argument(
        "--profile_wait",
        type=int,
        default=1,
        help="Profiler schedule: wait steps before warmup",
    )
    parser.add_argument(
        "--profile_warmup",
        type=int,
        default=1,
        help="Profiler schedule: warmup steps before active",
    )
    parser.add_argument(
        "--profile_active",
        type=int,
        default=5,
        help="Profiler schedule: number of active profiling steps",
    )

    parser.add_argument(
        "--debug_devices",
        action="store_true",
        help="Print detailed device/dtype info and fused optimizer group step diagnostics",
    )
    parser.add_argument(
        "--fused_hook_no_step",
        action="store_true",
        help="Do not call optimizer.step() inside post-accumulate hooks; step groups via fallback after backward",
    )
    parser.add_argument(
        "--fused_hook_debug_verbose",
        action="store_true",
        help="Verbose fused-hook logging (per-group counts/steps)",
    )
    parser.add_argument(
        "--fused_hook_no_cuda_sync",
        action="store_true",
        help="Skip torch.cuda.synchronize() in fused-hook debug paths",
    )
    parser.add_argument(
        "--fused_hook_timeout",
        type=int,
        default=60,
        help="Seconds before traceback dump if optimizer.step() stalls in hook",
    )

    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    
    # Handle Diff2Flow vs naive FM selection
    if args.naive_fm:
        args.use_diff2flow = False
    
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)

    train(args)
