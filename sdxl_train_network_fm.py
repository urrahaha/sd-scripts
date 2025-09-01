import argparse
from typing import List, Optional, Union, Any

import torch
from accelerate import Accelerator
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from diffusers.training_utils import compute_density_for_timestep_sampling

from library.device_utils import init_ipex

init_ipex()

from library import sdxl_model_util, sdxl_train_util, strategy_base, strategy_sd, strategy_sdxl, train_util
import train_network
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)
_warned_min_snr_gamma = False
_warned_vpred_flags = False


def _get_sigmas(
    noise_scheduler: FlowMatchEulerDiscreteScheduler,
    device: torch.device,
    timesteps: torch.Tensor,
    n_dim: int = 4,
    dtype: torch.dtype = torch.float32,
):
    """
    Vectorized lookup of sigma values for given timesteps from FlowMatch scheduler.
    Matches logic used in `sd-scripts/mosaic_train.py`.
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
    Diff2Flow wrapper that bridges diffusion and flow matching paradigms for SDXL UNet calls.
    It keeps inference in diffusion space while using FM-derived supervision.
    """

    def __init__(self, unet, parameterization: str = "v"):
        self.unet = unet
        self.parameterization = parameterization  # "v" or "eps"
        self.num_timesteps = 1000

        # Initialize diffusion-aligned schedule for rectified mappings
        self._setup_diffusion_schedule()

    def _setup_diffusion_schedule(self):
        """Setup diffusion schedule and stable precomputations."""
        import numpy as np
        from functools import partial

        # SDXL-like scaled linear schedule
        linear_start = 0.00085
        linear_end = 0.0120

        betas = self._make_beta_schedule("scaled_linear", self.num_timesteps, linear_start, linear_end)
        betas = self._enforce_zero_terminal_snr(betas)

        # Numerical stability clamps
        eps = 1e-12
        betas = np.clip(betas, eps, 1.0 - eps)
        alphas = np.clip(1.0 - betas, eps, 1.0 - eps)

        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod = np.clip(alphas_cumprod, eps, 1.0)
        alphas_cumprod_full = np.append(1.0, alphas_cumprod)
        alphas_cumprod_full = np.clip(alphas_cumprod_full, eps, 1.0)

        to_torch = partial(torch.tensor, dtype=torch.float32)

        # Minimal register_buffer shim (not a Module here)
        self.register_buffer = lambda name, tensor: setattr(self, name, tensor)

        # Base buffers
        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_full", to_torch(alphas_cumprod_full))

        # Stable square-roots and reciprocals
        sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = np.sqrt(np.clip(1.0 - alphas_cumprod, 0.0, 1.0))
        sqrt_alphas_cumprod_full = np.sqrt(alphas_cumprod_full)
        sqrt_one_minus_alphas_cumprod_full = np.sqrt(np.clip(1.0 - alphas_cumprod_full, 0.0, 1.0))

        inv_alphas_cumprod = 1.0 / np.clip(alphas_cumprod, eps, 1.0)
        sqrt_recip_alphas_cumprod = np.sqrt(inv_alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = np.sqrt(np.clip(inv_alphas_cumprod - 1.0, 0.0, None))

        self.register_buffer("sqrt_alphas_cumprod", to_torch(sqrt_alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", to_torch(sqrt_one_minus_alphas_cumprod))
        self.register_buffer("sqrt_alphas_cumprod_full", to_torch(sqrt_alphas_cumprod_full))
        self.register_buffer("sqrt_one_minus_alphas_cumprod_full", to_torch(sqrt_one_minus_alphas_cumprod_full))

        self.register_buffer("sqrt_recip_alphas_cumprod", to_torch(sqrt_recip_alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", to_torch(sqrt_recipm1_alphas_cumprod))

    def _make_beta_schedule(self, schedule, n_timestep, linear_start=1e-4, linear_end=2e-2):
        import numpy as np
        if schedule in ("linear", "scaled_linear"):
            return np.linspace(linear_start**0.5, linear_end**0.5, n_timestep, dtype=np.float64) ** 2
        raise ValueError(f"Unknown beta schedule: {schedule}")

    def _enforce_zero_terminal_snr(self, betas):
        import numpy as np
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)

        alphas_cumprod_final = alphas_cumprod[-1]
        if alphas_cumprod_final > 0:
            alphas_cumprod = alphas_cumprod / alphas_cumprod_final
            alphas = alphas_cumprod / np.concatenate([[1.0], alphas_cumprod[:-1]])
            betas = 1.0 - alphas
        return betas

    def convert_fm_t_to_dm_t(self, t: torch.Tensor) -> torch.Tensor:
        """
        Map FM time t in [0,1] to a continuous diffusion timestep index in [0, num_timesteps).
        """
        device = t.device
        rectified_alphas = (self.sqrt_alphas_cumprod_full + self.sqrt_one_minus_alphas_cumprod_full).to(device)

        # Reverse for searchsorted (ascending)
        rectified_alphas_rev = torch.flip(rectified_alphas, [0])

        # Normalize t onto rectified range for robust search
        t = t.to(rectified_alphas_rev.dtype)
        eps = torch.finfo(rectified_alphas_rev.dtype).eps
        t_min, t_max = rectified_alphas_rev[0] + eps, rectified_alphas_rev[-1] - eps
        t_clamped = t.clamp(min=t_min, max=t_max)

        right_idx = torch.searchsorted(rectified_alphas_rev, t_clamped, right=True)
        right_idx = right_idx.clamp(1, rectified_alphas_rev.shape[0] - 1)
        left_idx = right_idx - 1

        right_val = rectified_alphas_rev.gather(0, right_idx)
        left_val = rectified_alphas_rev.gather(0, left_idx)
        dm_t = left_idx + (t_clamped - left_val) / (right_val - left_val + 1e-8)
        return dm_t.clamp(0, self.num_timesteps - 1)

    def convert_fm_xt_to_dm_xt(self, fm_xt: torch.Tensor, fm_t: torch.Tensor) -> torch.Tensor:
        device = fm_xt.device
        scale = (self.sqrt_alphas_cumprod_full + self.sqrt_one_minus_alphas_cumprod_full).to(device)

        dm_t = self.convert_fm_t_to_dm_t(fm_t)
        dm_t_left = torch.floor(dm_t).long()
        dm_t_right = torch.ceil(dm_t).long()

        scale_left = scale[dm_t_left].view(-1, 1, 1, 1)
        scale_right = scale[dm_t_right].view(-1, 1, 1, 1)
        alpha = (dm_t - dm_t_left.float()).view(-1, 1, 1, 1)
        scale_t = scale_left + alpha * (scale_right - scale_left)

        dm_xt = fm_xt * scale_t
        return dm_xt.to(dtype=fm_xt.dtype)

    def predict_start_from_v(self, x_t, t, v):
        device = x_t.device
        sqrt_alphas = self.sqrt_alphas_cumprod.to(device)
        sqrt_one_minus_alphas = self.sqrt_one_minus_alphas_cumprod.to(device)
        return (
            sqrt_alphas[t.long()].view(-1, 1, 1, 1) * x_t
            - sqrt_one_minus_alphas[t.long()].view(-1, 1, 1, 1) * v
        )

    def predict_eps_from_v(self, x_t, t, v):
        device = x_t.device
        sqrt_alphas = self.sqrt_alphas_cumprod.to(device)
        sqrt_one_minus_alphas = self.sqrt_one_minus_alphas_cumprod.to(device)
        return (
            sqrt_alphas[t.long()].view(-1, 1, 1, 1) * v
            + sqrt_one_minus_alphas[t.long()].view(-1, 1, 1, 1) * x_t
        )

    def predict_start_from_eps(self, x_t, t, eps):
        device = x_t.device
        sqrt_recip_alphas = self.sqrt_recip_alphas_cumprod.to(device)
        sqrt_recipm1_alphas = self.sqrt_recipm1_alphas_cumprod.to(device)
        return (
            sqrt_recip_alphas[t.long()].view(-1, 1, 1, 1) * x_t
            - sqrt_recipm1_alphas[t.long()].view(-1, 1, 1, 1) * eps
        )

    def get_vector_field_from_diffusion_pred(self, diffusion_pred, dm_xt, dm_t):
        if self.parameterization == "v":
            x_0_pred = self.predict_start_from_v(dm_xt, dm_t, diffusion_pred)
            eps_pred = self.predict_eps_from_v(dm_xt, dm_t, diffusion_pred)
        elif self.parameterization == "eps":
            x_0_pred = self.predict_start_from_eps(dm_xt, dm_t, diffusion_pred)
            eps_pred = diffusion_pred
        else:
            raise ValueError(f"Unknown parameterization: {self.parameterization}")
        return x_0_pred - eps_pred

    def sample_vector_field(self, fm_xt, fm_t, text_embedding, vector_embedding):
        """Query UNet in diffusion space and convert to FM velocity field."""
        dm_t = self.convert_fm_t_to_dm_t(fm_t)
        dm_xt = self.convert_fm_xt_to_dm_xt(fm_xt, fm_t)

        # Match dtype between latent and conditioning to avoid cast at every step
        if dm_xt.dtype != vector_embedding.dtype:
            dm_xt = dm_xt.to(dtype=vector_embedding.dtype)
        dm_t_long = dm_t.long()

        diffusion_pred = self.unet(dm_xt, dm_t_long, text_embedding, vector_embedding)

        if torch.isnan(diffusion_pred).any():
            logger.warning("NaN detected in diffusion prediction, replacing with zeros")
            diffusion_pred = torch.nan_to_num(diffusion_pred, 0.0)

        return self.get_vector_field_from_diffusion_pred(diffusion_pred, dm_xt, dm_t)


def _maybe_patch_dora_for_speed(args):
    """
    Optionally monkey-patch LyCORIS LoConModule's DoRA path to a faster, lower-memory variant:
    - Caches the denominator (per-channel norm of weight + diff) and recomputes every N forwards
    - Optionally computes the denominator under no_grad to reduce autograd state

    Controlled by CLI flags:
      --dora_fast (enable), --dora_cache_interval N (default 1 => no caching), --dora_no_grad_norm
    """
    enable = getattr(args, "dora_fast", False) or getattr(args, "dora_cache_interval", 1) > 1 or getattr(args, "dora_no_grad_norm", False)
    if not enable:
        return
    try:
        from lycoris.modules.locon import LoConModule  # type: ignore
    except Exception as e:  # pragma: no cover - optional dep
        logger.warning(f"DoRA fast patch skipped: cannot import LyCORIS LoConModule ({e})")
        return

    interval = max(1, int(getattr(args, "dora_cache_interval", 1)))
    use_no_grad = bool(getattr(args, "dora_no_grad_norm", False))

    def _apply_weight_decompose_fast(self, weight, multiplier=1):  # noqa: N802 (match original name)
        # Recompute cached denominator based on interval while training; always keep device/dtype in sync
        step = int(getattr(self, "_dora_step", 0))
        cached = getattr(self, "_dora_cached_denom", None)
        need_recompute = True
        if self.training and interval > 1 and cached is not None:
            need_recompute = (step % interval) == 0 or cached.device != weight.device or cached.dtype != weight.dtype
        elif cached is not None:
            # not training or interval==1: reuse if compatible
            need_recompute = cached.device != weight.device or cached.dtype != weight.dtype

        if need_recompute:
            if use_no_grad:
                with torch.no_grad():
                    denom = torch.linalg.vector_norm(weight, dim=self._reduce_dims, keepdim=True)
            else:
                denom = torch.linalg.vector_norm(weight, dim=self._reduce_dims, keepdim=True)
            denom = denom.clamp_min(torch.finfo(weight.dtype).eps)
            self._dora_cached_denom = denom
        else:
            denom = cached

        # advance local step counter
        self._dora_step = step + 1

        scale = self.dora_scale.to(weight.device, dtype=weight.dtype) / denom
        if multiplier != 1:
            scale = multiplier * (scale - 1) + 1
        return weight * scale

    # Install once per process
    if getattr(LoConModule, "_dora_fast_patched", False) is not True:
        LoConModule.apply_weight_decompose = _apply_weight_decompose_fast  # type: ignore[attr-defined]
        LoConModule._dora_fast_patched = True  # type: ignore[attr-defined]
        logger.info(
            f"Enabled DoRA fast mode for LyCORIS LoConModule (cache_interval={interval}, no_grad_norm={use_no_grad})"
        )


class SdxlNetworkTrainerFM(train_network.NetworkTrainer):
    """
    SDXL LoRA trainer variant using Flow Matching loss.
    - Uses FlowMatchEulerDiscreteScheduler.
    - Samples timesteps using logit-normal density.
    - Constructs z_t = (1 - sigma) * x + sigma * eps and target = eps - x.
    - Calls UNet with SDXL conditioning.
    """

    def __init__(self):
        super().__init__()
        self.vae_scale_factor = sdxl_model_util.VAE_SCALE_FACTOR
        self.is_sdxl = True
        self.diff2flow_wrapper = None

    # region SDXL-specific overrides
    def assert_extra_args(
        self,
        args,
        train_dataset_group: Union[train_util.DatasetGroup, train_util.MinimalDataset],
        val_dataset_group: Optional[train_util.DatasetGroup],
    ):
        sdxl_train_util.verify_sdxl_training_args(args)
        train_dataset_group.verify_bucket_reso_steps(32)
        if val_dataset_group is not None:
            val_dataset_group.verify_bucket_reso_steps(32)

    def load_target_model(self, args, weight_dtype, accelerator):
        (
            load_stable_diffusion_format,
            text_encoder1,
            text_encoder2,
            vae,
            unet,
            logit_scale,
            ckpt_info,
        ) = sdxl_train_util.load_target_model(
            args, accelerator, sdxl_model_util.MODEL_VERSION_SDXL_BASE_V1_0, weight_dtype
        )

        self.load_stable_diffusion_format = load_stable_diffusion_format
        self.logit_scale = logit_scale
        self.ckpt_info = ckpt_info

        # Enable memory efficient attention etc.
        train_util.replace_unet_modules(unet, args.mem_eff_attn, args.xformers, args.sdpa)

        if torch.__version__ >= "2.0.0":
            vae.set_use_memory_efficient_attention_xformers(args.xformers)

        return sdxl_model_util.MODEL_VERSION_SDXL_BASE_V1_0, [text_encoder1, text_encoder2], vae, unet

    def get_tokenize_strategy(self, args):
        return strategy_sdxl.SdxlTokenizeStrategy(args.max_token_length, args.tokenizer_cache_dir)

    def get_tokenizers(self, tokenize_strategy: strategy_sdxl.SdxlTokenizeStrategy):
        return [tokenize_strategy.tokenizer1, tokenize_strategy.tokenizer2]

    def get_latents_caching_strategy(self, args):
        return strategy_sd.SdSdxlLatentsCachingStrategy(False, args.cache_latents_to_disk, args.vae_batch_size, args.skip_cache_check)

    def get_text_encoding_strategy(self, args):
        return strategy_sdxl.SdxlTextEncodingStrategy()

    def get_models_for_text_encoding(self, args, accelerator, text_encoders):
        # SDXL uses wrapped + unwrapped for some paths in utilities
        return text_encoders + [accelerator.unwrap_model(text_encoders[-1])]

    def get_text_encoder_outputs_caching_strategy(self, args):
        if args.cache_text_encoder_outputs:
            return strategy_sdxl.SdxlTextEncoderOutputsCachingStrategy(
                args.cache_text_encoder_outputs_to_disk, None, args.skip_cache_check, is_weighted=args.weighted_captions
            )
        return None

    def cache_text_encoder_outputs_if_needed(
        self, args, accelerator: Accelerator, unet, vae, text_encoders, dataset: train_util.DatasetGroup, weight_dtype
    ):
        if args.cache_text_encoder_outputs:
            if not args.lowram:
                logger.info("move vae and unet to cpu to save memory")
                org_vae_device = vae.device
                org_unet_device = unet.device
                vae.to("cpu")
                unet.to("cpu")
            text_encoders[0].to(accelerator.device, dtype=weight_dtype)
            text_encoders[1].to(accelerator.device, dtype=weight_dtype)
            with accelerator.autocast():
                dataset.new_cache_text_encoder_outputs(text_encoders + [accelerator.unwrap_model(text_encoders[-1])], accelerator)
            accelerator.wait_for_everyone()
            text_encoders[0].to("cpu", dtype=torch.float32)
            text_encoders[1].to("cpu", dtype=torch.float32)
            if not args.lowram:
                vae.to(org_vae_device)
                unet.to(org_unet_device)
        else:
            text_encoders[0].to(accelerator.device, dtype=weight_dtype)
            text_encoders[1].to(accelerator.device, dtype=weight_dtype)

    # endregion

    # region Flow Matching overrides
    def get_noise_scheduler(self, args: argparse.Namespace, device: torch.device) -> Any:
        """Use FlowMatch Euler discrete scheduler for training."""
        noise_scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0)
        return noise_scheduler

    def call_unet(
        self,
        args,
        accelerator,
        unet,
        noisy_latents,
        timesteps,
        text_conds,
        batch,
        weight_dtype,
        indices: Optional[List[int]] = None,
    ):
        noisy_latents = noisy_latents.to(weight_dtype)

        # SDXL size/time embeddings
        orig_size = batch["original_sizes_hw"]
        crop_size = batch["crop_top_lefts"]
        target_size = batch["target_sizes_hw"]
        embs = sdxl_train_util.get_size_embeddings(orig_size, crop_size, target_size, accelerator.device).to(weight_dtype)

        # concat embeddings
        encoder_hidden_states1, encoder_hidden_states2, pool2 = text_conds
        vector_embedding = torch.cat([pool2, embs], dim=1).to(weight_dtype)
        text_embedding = torch.cat([encoder_hidden_states1, encoder_hidden_states2], dim=2).to(weight_dtype)

        if indices is not None and len(indices) > 0:
            noisy_latents = noisy_latents[indices]
            timesteps = timesteps[indices]
            text_embedding = text_embedding[indices]
            vector_embedding = vector_embedding[indices]

        noise_pred = unet(noisy_latents, timesteps, text_embedding, vector_embedding)
        return noise_pred

    def get_noise_pred_and_target(
        self,
        args,
        accelerator,
        noise_scheduler: FlowMatchEulerDiscreteScheduler,
        latents,
        batch,
        text_encoder_conds,
        unet,
        network,
        weight_dtype,
        train_unet,
        is_train=True,
    ):
        bsz = latents.shape[0]
        device = latents.device

        if getattr(args, "use_diff2flow", False):
            # Diff2Flow: sample FM time uniformly, build FM trajectory and velocity target
            fm_t = torch.rand(bsz, device=device, dtype=latents.dtype)
            noise = torch.randn_like(latents)
            fm_xt = (1.0 - fm_t.view(-1, 1, 1, 1)) * noise + fm_t.view(-1, 1, 1, 1) * latents
            target = latents - noise

            # SDXL size/time embeddings
            orig_size = batch["original_sizes_hw"]
            crop_size = batch["crop_top_lefts"]
            target_size = batch["target_sizes_hw"]
            embs = sdxl_train_util.get_size_embeddings(
                orig_size, crop_size, target_size, accelerator.device
            ).to(device=accelerator.device, dtype=weight_dtype)

            encoder_hidden_states1, encoder_hidden_states2, pool2 = text_encoder_conds
            vector_embedding = torch.cat([pool2, embs], dim=1).to(device=accelerator.device, dtype=weight_dtype)
            text_embedding = torch.cat([encoder_hidden_states1, encoder_hidden_states2], dim=2).to(
                device=accelerator.device, dtype=weight_dtype
            )

            # Init wrapper once (respect parameterization flag)
            if self.diff2flow_wrapper is None:
                param = getattr(args, "d2f_param", "v")
                self.diff2flow_wrapper = Diff2FlowWrapper(unet, parameterization=param)

            if args.gradient_checkpointing:
                fm_xt.requires_grad_(True)
                text_embedding.requires_grad_(True)
                vector_embedding.requires_grad_(True)

            with torch.set_grad_enabled(is_train), accelerator.autocast():
                noise_pred = self.diff2flow_wrapper.sample_vector_field(
                    fm_xt, fm_t, text_embedding, vector_embedding
                )

            # Dummy integer timesteps for API compatibility (not used by post_process_loss here)
            dummy_ts = torch.zeros(bsz, dtype=torch.int64, device=device)
            return noise_pred, target, dummy_ts, None

        # Naive FM (original implementation)
        u = compute_density_for_timestep_sampling(
            weighting_scheme="logit_normal",
            batch_size=bsz,
            logit_mean=0.0,
            logit_std=1.0,
            mode_scale=1.29,
        )
        indices = (u * noise_scheduler.config.num_train_timesteps).long().clamp(
            min=0, max=noise_scheduler.config.num_train_timesteps - 1
        )
        timesteps = noise_scheduler.timesteps[indices].to(device=device)

        sigmas = _get_sigmas(
            noise_scheduler, device, timesteps, n_dim=latents.ndim, dtype=latents.dtype
        )
        noise = torch.randn(latents.size(), device=device, dtype=latents.dtype)
        noisy_latents = (1.0 - sigmas) * latents + sigmas * noise

        if args.gradient_checkpointing:
            for x in noisy_latents:
                x.requires_grad_(True)
            for t in text_encoder_conds:
                t.requires_grad_(True)

        with torch.set_grad_enabled(is_train), accelerator.autocast():
            noise_pred = self.call_unet(
                args,
                accelerator,
                unet,
                noisy_latents.requires_grad_(train_unet),
                timesteps,
                text_encoder_conds,
                batch,
                weight_dtype,
            )

        target = noise - latents
        return noise_pred, target, timesteps, None

    def post_process_loss(self, loss, args, timesteps: torch.IntTensor, noise_scheduler) -> torch.FloatTensor:
        # For Flow Matching, we do NOT apply SNR weighting or v-pred/v-like/debiased extras.
        # Those post-process steps are designed for DDPM/DDIM training and may rely on
        # scheduler attributes (e.g., all_snr) not present in FlowMatch schedulers.
        # If the user passes such flags (e.g., min_snr_gamma), warn and skip.
        global _warned_min_snr_gamma, _warned_vpred_flags
        if getattr(args, "min_snr_gamma", 0) and not _warned_min_snr_gamma:
            logger.warning(
                "min_snr_gamma is not applicable to Flow Matching; skipping SNR weighting."
            )
            _warned_min_snr_gamma = True
        if (
            getattr(args, "scale_v_pred_loss_like_noise_pred", False)
            or getattr(args, "v_pred_like_loss", None)
            or getattr(args, "debiased_estimation_loss", None)
        ) and not _warned_vpred_flags:
            logger.warning(
                "v-pred-like loss scaling or debiased estimation flags are not applied in Flow Matching; skipping."
            )
            _warned_vpred_flags = True
        return loss

    # endregion

    def get_text_cond(self, args, accelerator, batch, tokenizers, text_encoders, weight_dtype):
        # Reuse sdxl_train_network flow for obtaining SDXL text conds
        if "text_encoder_outputs1_list" not in batch or batch["text_encoder_outputs1_list"] is None:
            input_ids1 = batch["input_ids"]
            input_ids2 = batch["input_ids2"]
            with torch.enable_grad():
                input_ids1 = input_ids1.to(accelerator.device)
                input_ids2 = input_ids2.to(accelerator.device)
                encoder_hidden_states1, encoder_hidden_states2, pool2 = train_util.get_hidden_states_sdxl(
                    args.max_token_length,
                    input_ids1,
                    input_ids2,
                    tokenizers[0],
                    tokenizers[1],
                    text_encoders[0],
                    text_encoders[1],
                    None if not args.full_fp16 else weight_dtype,
                    accelerator=accelerator,
                )
        else:
            encoder_hidden_states1 = batch["text_encoder_outputs1_list"].to(accelerator.device).to(weight_dtype)
            encoder_hidden_states2 = batch["text_encoder_outputs2_list"].to(accelerator.device).to(weight_dtype)
            pool2 = batch["text_encoder_pool2_list"].to(accelerator.device).to(weight_dtype)
        return encoder_hidden_states1, encoder_hidden_states2, pool2

    def post_process_network(self, args, accelerator, network, text_encoders, unet):
        # Ensure the Diff2Flow wrapper is recreated after accelerator.prepare so it captures
        # the accelerator-prepared UNet handle, avoiding stale references.
        self.diff2flow_wrapper = None
        # Optionally speed up DoRA (LyCORIS LoConModule) for lower memory and faster training.
        _maybe_patch_dora_for_speed(args)

    def sample_images(self, accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, unet):
        sdxl_train_util.sample_images(accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, unet)


def setup_parser() -> argparse.ArgumentParser:
    parser = train_network.setup_parser()
    sdxl_train_util.add_sdxl_training_arguments(parser)
    # Diff2Flow options
    parser.add_argument(
        "--use_diff2flow",
        action="store_true",
        help="Use Diff2Flow (FM supervision over diffusion UNet) instead of naive FM pipeline",
    )
    parser.add_argument(
        "--d2f_param",
        type=str,
        choices=["v", "eps"],
        default="v",
        help="UNet parameterization assumed by Diff2Flow wrapper",
    )
    # DoRA (LyCORIS LoConModule) speed/memory optimization options
    parser.add_argument(
        "--dora_fast",
        action="store_true",
        help="Enable faster, lower-memory DoRA by caching per-channel norms and optionally computing them under no_grad.",
    )
    parser.add_argument(
        "--dora_cache_interval",
        type=int,
        default=1,
        help="Recompute DoRA denominator (per-channel norm) every N forwards (>=1). Higher values trade accuracy for speed/memory.",
    )
    parser.add_argument(
        "--dora_no_grad_norm",
        action="store_true",
        help="Compute DoRA denominator under torch.no_grad to reduce autograd memory (approximate).",
    )
    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)

    trainer = SdxlNetworkTrainerFM()
    trainer.train(args)
