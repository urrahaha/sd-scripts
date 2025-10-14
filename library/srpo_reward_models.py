"""
SRPO Reward Models for Direct Preference Alignment
Based on: https://github.com/Tencent-Hunyuan/SRPO

Supports HPS-v2.1, PickScore, and CLIP reward models for SDXL training.
"""

import torch
import torch.nn as nn
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    from PIL import Image
    BICUBIC = Image.BICUBIC

from transformers import AutoProcessor, AutoModel
import logging

logger = logging.getLogger(__name__)


def get_random_cg_oily_adjective(index=0):
    """Get control words for negative preference (flat/oily textures)"""
    cg_oily_adjectives = [
        "Concept art",
        "Painting", 
        "Anime",
        "Flat",
        "Oil"
    ]
    return cg_oily_adjectives[index % len(cg_oily_adjectives)]


def get_random_realism_adjective(index=0):
    """Get control words for positive preference (realistic/detailed)"""
    realism_adjectives = [
        "Natural-lighting",
        "Detail",
        "Detailed",
        "Real"
    ]
    return realism_adjectives[index % len(realism_adjectives)]


class CLIPRewardModel(nn.Module):
    """CLIP-based reward model (supports PickScore mode)"""
    
    def __init__(self, is_pickscore=False, device="cuda", dtype=torch.float32):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.is_pickscore = is_pickscore
        
        # Model paths - can be overridden
        self.processor_path = "./models/reward_models/clip"
        self.model_path = "./models/reward_models/pickscore" if is_pickscore else "./models/reward_models/clip"
        
        try:
            self.processor = AutoProcessor.from_pretrained(self.processor_path)
            self.model = AutoModel.from_pretrained(self.model_path).eval().to(device)
            self.model = self.model.to(dtype=dtype)
        except Exception as e:
            logger.error(f"Failed to load CLIP reward model: {e}")
            logger.info("Please download models using: huggingface-cli download")
            raise
        
        # Image preprocessing
        image_mean = (0.48145466, 0.4578275, 0.40821073)
        image_std = (0.26862954, 0.26130258, 0.27577711)
        crop_size = 224
        resize_size = 224
        
        self.v_pre = Compose([
            Resize(resize_size, interpolation=BICUBIC),
            CenterCrop(crop_size),
            Normalize(std=image_std, mean=image_mean),
        ])
    
    def srp_cfg(self, prompt, neg_prompt, image_inputs, k):
        """
        Style Reward Preference with CFG-like formulation
        Reward = (1+k)*pos_reward - neg_reward
        """
        image_inputs = self.v_pre(image_inputs)
        
        # Process text prompts
        text_inputs = self.processor(
            text=prompt,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        text_inputs = {k: v.to(device=self.device) for k, v in text_inputs.items()}
        
        neg_text_input = self.processor(
            text=neg_prompt,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        neg_text_input = {k: v.to(device=self.device) for k, v in neg_text_input.items()}
        
        # Extract features
        image_embs = self.model.get_image_features(pixel_values=image_inputs)
        image_embs = image_embs / image_embs.norm(p=2, dim=-1, keepdim=True)
        
        text_embs = self.model.get_text_features(**text_inputs)
        text_embs = text_embs / text_embs.norm(p=2, dim=-1, keepdim=True)
        
        text_embs_neg = self.model.get_text_features(**neg_text_input)
        text_embs_neg = text_embs_neg / text_embs_neg.norm(p=2, dim=-1, keepdim=True)
        
        logit_scale = self.model.logit_scale.exp()
        
        # Compute reward: (1+k)*pos - neg
        scores = logit_scale * ((k + 1) * text_embs - text_embs_neg) @ image_embs.T
        scores = scores.diag()
        
        # Scale to be comparable with HPS
        scores = scores / 20
        return scores


class HPSRewardModel(nn.Module):
    """HPS-v2.1 reward model for human preference alignment"""
    
    def __init__(self, device="cuda", dtype=torch.float32):
        super().__init__()
        self.device = device
        self.dtype = dtype
        
        # Build reward model
        hpsv2_model, hpsv2_token, hpsv2_pre = self.build_reward_model()
        self.model = hpsv2_model.to(dtype=dtype)
        self.token = hpsv2_token
        
        # Differentiable preprocessor
        image_mean = (0.48145466, 0.4578275, 0.40821073)
        image_std = (0.26862954, 0.26130258, 0.27577711)
        crop_size = 224
        resize_size = 224
        
        self.vis_pre = Compose([
            Resize(resize_size, interpolation=BICUBIC),
            CenterCrop(crop_size),
            Normalize(std=image_std, mean=image_mean),
        ])
        
    def build_reward_model(self):
        """Load HPS-v2.1 model and tokenizer"""
        try:
            from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
        except ImportError:
            logger.error("HPS-v2 package not found. Install with: pip install hpsv2")
            raise
        
        model, preprocess_train, preprocess_val = create_model_and_transforms(
            'ViT-H-14',
            'laion2B-s32B-b79K',
            precision='amp',
            device=self.device,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            light_augmentation=True,
            aug_cfg={},
            output_dict=True,
            with_score_predictor=False,
            with_region_predictor=False
        )
        
        # Load checkpoint
        checkpoint_path = './models/reward_models/hps_v2.1/HPS_v2.1_compressed.pt'
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            model.load_state_dict(checkpoint['state_dict'])
        except Exception as e:
            logger.error(f"Failed to load HPS checkpoint from {checkpoint_path}: {e}")
            logger.info("Download with: huggingface-cli download xswu/HPSv2 HPS_v2.1_compressed.pt")
            raise
        
        text_processor = get_tokenizer('ViT-H-14')
        reward_model = model.to(self.device)
        reward_model.eval()
        
        return reward_model, text_processor, preprocess_train
    
    def srp_cfg(self, prompt, neg_prompt, images, k):
        """
        Style Reward Preference with CFG-like formulation
        Reward = (1+k)*pos_reward - neg_reward
        """
        image = self.vis_pre(images.squeeze(0)).unsqueeze(0).to(device=self.device, non_blocking=True)
        text = self.token(prompt).to(device=self.device, non_blocking=True)
        neg_text = self.token(neg_prompt).to(device=self.device, non_blocking=True)
        
        with torch.cuda.amp.autocast():
            # Extract features
            image_features = self.model.encode_image(image, normalize=True)
            text_features = self.model.encode_text(text, normalize=True)
            text_features_neg = self.model.encode_text(neg_text, normalize=True)
            
            # Compute reward: (1+k)*pos - neg
            logits_per_image = image_features @ ((1 + k) * text_features.T - text_features_neg.T)
            hps_score = torch.diagonal(logits_per_image)
        
        return hps_score
    
    def srp(self, prompt, images, k):
        """Simple SRP without negative prompt"""
        image = self.vis_pre(images.squeeze(0)).unsqueeze(0).to(device=self.device, non_blocking=True)
        text = self.token(prompt).to(device=self.device, non_blocking=True)
        
        with torch.cuda.amp.autocast():
            image_features = self.model.encode_image(image, normalize=True)
            text_features = self.model.encode_text(text, normalize=True)
            logits_per_image = image_features @ (k * text_features.T)
            hps_score = torch.diagonal(logits_per_image)
        
        return hps_score


def build_reward_model(reward_model_type, device="cuda", dtype=torch.float32):
    """
    Factory function to build reward models
    
    Args:
        reward_model_type: "HPS", "PickScore", or "CLIP"
        device: torch device
        dtype: torch dtype
    
    Returns:
        Reward model instance
    """
    reward_model_type = reward_model_type.upper()
    
    if reward_model_type == "HPS":
        logger.info("Initializing HPS-v2.1 reward model...")
        return HPSRewardModel(device=device, dtype=dtype)
    elif reward_model_type == "PICKSCORE":
        logger.info("Initializing PickScore reward model...")
        return CLIPRewardModel(is_pickscore=True, device=device, dtype=dtype)
    elif reward_model_type == "CLIP":
        logger.info("Initializing CLIP reward model...")
        return CLIPRewardModel(is_pickscore=False, device=device, dtype=dtype)
    else:
        raise ValueError(f"Unsupported reward model: {reward_model_type}. Choose from: HPS, PickScore, CLIP")
