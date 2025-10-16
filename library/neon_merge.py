"""
Neon Merge Implementation

Applies negative extrapolation: θ_neon = (1+w)·θ_base - w·θ_aux
"""

import torch
from pathlib import Path
import logging
from safetensors.torch import save_file, load_file

logger = logging.getLogger(__name__)


def neon_merge(
    base_model_path: str,
    aux_model_path: str,
    output_path: str,
    extrapolation_weight: float = 0.3,
    save_format: str = "safetensors",
):
    """
    Apply Neon merge to create improved model.
    
    Formula: θ_neon = (1+w)·θ_base - w·θ_aux
    
    Where:
    - θ_base: Original trained model (before synthetic training)
    - θ_aux: Auxiliary model (after synthetic training)
    - w: Extrapolation weight (typically 0.1-0.5)
    
    Args:
        base_model_path: Path to base model (pre-synthetic)
        aux_model_path: Path to auxiliary model (post-synthetic)
        output_path: Where to save merged model
        extrapolation_weight: Neon extrapolation weight (default: 0.3)
        save_format: Output format (safetensors, ckpt, pt)
    
    Returns:
        Path to saved merged model
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("Applying Neon Merge")
    logger.info("=" * 60)
    logger.info(f"Base model: {base_model_path}")
    logger.info(f"Auxiliary model: {aux_model_path}")
    logger.info(f"Extrapolation weight (w): {extrapolation_weight}")
    logger.info(f"Formula: θ_neon = (1+{extrapolation_weight})·θ_base - {extrapolation_weight}·θ_aux")
    logger.info("")
    
    # Load models
    logger.info("Loading base model...")
    if base_model_path.endswith('.safetensors'):
        base_state_dict = load_file(base_model_path)
    else:
        base_state_dict = torch.load(base_model_path, map_location='cpu')
        if 'state_dict' in base_state_dict:
            base_state_dict = base_state_dict['state_dict']
    
    logger.info("Loading auxiliary model...")
    if aux_model_path.endswith('.safetensors'):
        aux_state_dict = load_file(aux_model_path)
    else:
        aux_state_dict = torch.load(aux_model_path, map_location='cpu')
        if 'state_dict' in aux_state_dict:
            aux_state_dict = aux_state_dict['state_dict']
    
    # Verify keys match
    base_keys = set(base_state_dict.keys())
    aux_keys = set(aux_state_dict.keys())
    
    if base_keys != aux_keys:
        logger.warning("Key mismatch between base and auxiliary models!")
        logger.warning(f"Base has {len(base_keys)} keys, aux has {len(aux_keys)} keys")
        
        # Use intersection for safety
        common_keys = base_keys & aux_keys
        logger.info(f"Using {len(common_keys)} common keys")
    else:
        common_keys = base_keys
        logger.info(f"✓ Models have {len(common_keys)} matching keys")
    
    # Apply Neon formula
    logger.info("")
    logger.info("Applying extrapolation formula...")
    neon_state_dict = {}
    
    w = extrapolation_weight
    
    for key in common_keys:
        base_param = base_state_dict[key]
        aux_param = aux_state_dict[key]
        
        # θ_neon = (1+w)·θ_base - w·θ_aux
        neon_param = (1 + w) * base_param - w * aux_param
        neon_state_dict[key] = neon_param
    
    logger.info(f"✓ Merged {len(neon_state_dict)} parameters")
    
    # Save merged model
    logger.info("")
    logger.info(f"Saving Neon-merged model to: {output_path}")
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if save_format == "safetensors" or output_path.suffix == ".safetensors":
        # Save as safetensors
        save_file(neon_state_dict, str(output_path))
    else:
        # Save as PyTorch checkpoint
        torch.save(neon_state_dict, str(output_path))
    
    logger.info("=" * 60)
    logger.info("✅ Neon Merge Complete!")
    logger.info("=" * 60)
    logger.info(f"Merged model saved: {output_path}")
    logger.info("")
    
    return str(output_path)


def verify_neon_merge(
    base_path: str,
    aux_path: str,
    neon_path: str,
    extrapolation_weight: float,
    sample_keys: int = 5,
):
    """
    Verify Neon merge was applied correctly.
    
    Checks that formula was applied: θ_neon = (1+w)·θ_base - w·θ_aux
    
    Args:
        base_path: Path to base model
        aux_path: Path to auxiliary model
        neon_path: Path to Neon-merged model
        extrapolation_weight: Expected extrapolation weight
        sample_keys: Number of keys to sample and verify
    
    Returns:
        True if verification passed, False otherwise
    """
    import random
    
    logger.info("Verifying Neon merge...")
    
    # Load models
    base_sd = load_file(base_path) if base_path.endswith('.safetensors') else torch.load(base_path, map_location='cpu')
    aux_sd = load_file(aux_path) if aux_path.endswith('.safetensors') else torch.load(aux_path, map_location='cpu')
    neon_sd = load_file(neon_path) if neon_path.endswith('.safetensors') else torch.load(neon_path, map_location='cpu')
    
    # Handle nested state_dict
    for sd in [base_sd, aux_sd, neon_sd]:
        if isinstance(sd, dict) and 'state_dict' in sd:
            sd = sd['state_dict']
    
    # Sample random keys
    common_keys = list(set(base_sd.keys()) & set(aux_sd.keys()) & set(neon_sd.keys()))
    sample_keys_list = random.sample(common_keys, min(sample_keys, len(common_keys)))
    
    w = extrapolation_weight
    all_match = True
    
    for key in sample_keys_list:
        base_param = base_sd[key]
        aux_param = aux_sd[key]
        neon_param = neon_sd[key]
        
        # Calculate expected
        expected = (1 + w) * base_param - w * aux_param
        
        # Check if close
        if not torch.allclose(neon_param, expected, rtol=1e-5, atol=1e-8):
            logger.error(f"Mismatch in key: {key}")
            all_match = False
        else:
            logger.info(f"  ✓ {key}: Verified")
    
    if all_match:
        logger.info("✅ Verification passed!")
        return True
    else:
        logger.error("❌ Verification failed!")
        return False
