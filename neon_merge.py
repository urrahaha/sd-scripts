#!/usr/bin/env python3
"""
Standalone Neon Merge Tool for LoRA/Model Weights

Usage:
    python neon_merge.py \
        --base_checkpoint base.safetensors \
        --aux_checkpoint aux.safetensors \
        --output neon_merged.safetensors \
        --extrapolation_weight 0.3

Based on Neon: Negative Extrapolation from Self-Training
Paper: https://arxiv.org/abs/2510.03597
"""

import argparse
import sys
import logging
from pathlib import Path

# Add library to path
sys.path.insert(0, str(Path(__file__).parent / "library"))

from library.neon_train_utils import neon_merge_lora

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Merge two LoRA checkpoints using Neon negative extrapolation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic merge with default weight
    python neon_merge.py \
        --base base.safetensors \
        --aux aux.safetensors \
        --output neon.safetensors

    # Aggressive extrapolation
    python neon_merge.py \
        --base base.safetensors \
        --aux aux.safetensors \
        --output neon_strong.safetensors \
        --extrapolation_weight 0.5

    # Conservative merge
    python neon_merge.py \
        --base base.safetensors \
        --aux aux.safetensors \
        --output neon_conservative.safetensors \
        --extrapolation_weight 0.1

Formula:
    θ_neon = (1+w)·θ_base - w·θ_aux
    
Where:
    - θ_base: Reference model (trained on real data)
    - θ_aux: Auxiliary model (fine-tuned on synthetic data)
    - w: Extrapolation weight (controls strength)
        """
    )
    
    parser.add_argument(
        "--base_checkpoint",
        "--base",
        type=str,
        required=True,
        help="Path to base/reference LoRA checkpoint (trained on real data)",
    )
    
    parser.add_argument(
        "--aux_checkpoint",
        "--aux",
        type=str,
        required=True,
        help="Path to auxiliary LoRA checkpoint (fine-tuned on synthetic data)",
    )
    
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Path for output Neon-merged checkpoint",
    )
    
    parser.add_argument(
        "--extrapolation_weight",
        "--weight",
        "-w",
        type=float,
        default=0.3,
        help="Neon extrapolation weight. Range: 0.1-0.5. Higher = stronger extrapolation (default: 0.3)",
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for merge computation (default: cuda)",
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    base_path = Path(args.base_checkpoint)
    aux_path = Path(args.aux_checkpoint)
    output_path = Path(args.output)
    
    if not base_path.exists():
        logger.error(f"Base checkpoint not found: {base_path}")
        sys.exit(1)
    
    if not aux_path.exists():
        logger.error(f"Auxiliary checkpoint not found: {aux_path}")
        sys.exit(1)
    
    if args.extrapolation_weight <= 0 or args.extrapolation_weight >= 1:
        logger.warning(f"Unusual extrapolation weight: {args.extrapolation_weight}")
        logger.warning("Typical range is 0.1-0.5. Proceed with caution!")
    
    # Display merge info
    logger.info("=" * 60)
    logger.info("Neon Merge Configuration")
    logger.info("=" * 60)
    logger.info(f"Base checkpoint:    {base_path}")
    logger.info(f"Aux checkpoint:     {aux_path}")
    logger.info(f"Output:             {output_path}")
    logger.info(f"Extrapolation (w):  {args.extrapolation_weight}")
    logger.info(f"Device:             {args.device}")
    logger.info("=" * 60)
    
    # Perform merge
    try:
        import torch
        device = torch.device(args.device)
        
        neon_merge_lora(
            base_lora_path=str(base_path),
            aux_lora_path=str(aux_path),
            output_path=str(output_path),
            w=args.extrapolation_weight,
            device=device,
        )
        
        logger.info("=" * 60)
        logger.info("✅ Neon merge completed successfully!")
        logger.info(f"Output saved to: {output_path}")
        logger.info("=" * 60)
        logger.info("")
        logger.info("Next steps:")
        logger.info("1. Test the merged model in your SD/SDXL workflow")
        logger.info("2. Compare quality vs base checkpoint")
        logger.info("3. If needed, adjust --extrapolation_weight and re-merge")
        logger.info("")
        logger.info("Typical weight ranges:")
        logger.info("  - 0.1-0.2: Conservative (safer)")
        logger.info("  - 0.3:     Recommended default")
        logger.info("  - 0.4-0.5: Aggressive (higher risk/reward)")
        
    except Exception as e:
        logger.error(f"Merge failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
