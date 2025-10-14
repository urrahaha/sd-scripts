#!/usr/bin/env python3
"""
Test script for Neon synthetic image generation pipeline

Tests the complete workflow:
1. Analyze dataset structure
2. Build generation plan
3. Generate synthetic images (test mode with placeholders)
4. Verify output structure and captions

Usage:
    # Test with placeholder images (no model loading)
    python test_neon_generation.py \
        --train_dir ./my_dataset \
        --output ./test_synthetic \
        --test_mode

    # Test with actual model (requires trained LoRA)
    python test_neon_generation.py \
        --train_dir ./my_dataset \
        --output ./test_synthetic \
        --model_path ./trained_lora.safetensors \
        --base_model "stabilityai/stable-diffusion-xl-base-1.0"
"""

import sys
import argparse
from pathlib import Path

# Add library to path
sys.path.insert(0, str(Path(__file__).parent / "library"))


class Args:
    """Mock args object for testing"""
    def __init__(self, output_dir, seed=None):
        self.output_dir = output_dir
        self.seed = seed
        self.neon_enable = True
        self.neon_synthetic_dataset_dir = None
        self.neon_guidance_scale = 7.5
        self.neon_inference_steps = 28


def test_structure_and_plan(train_dir: str, output_dir: str, percent: float = 100.0):
    """Test dataset structure replication and generation plan building"""
    from library.neon_train_utils import replicate_dataset_structure
    
    print("\n" + "=" * 60)
    print("PHASE 1: Structure Replication & Plan Building")
    print("=" * 60)
    
    synthetic_path, generation_plan = replicate_dataset_structure(
        train_data_dir=train_dir,
        output_dir=output_dir,
        synthetic_percent=percent,
        reuse_existing=True,
    )
    
    print("\n✓ Structure replicated successfully")
    print(f"  Synthetic dataset at: {synthetic_path}")
    
    # Analyze generation plan
    total_tasks = sum(len(info.get('generation_tasks', [])) for info in generation_plan.values())
    print(f"\n✓ Generation plan created")
    print(f"  Total generation tasks: {total_tasks}")
    
    if total_tasks > 0:
        print(f"\n  Sample generation tasks:")
        for subset_name, info in generation_plan.items():
            tasks = info.get('generation_tasks', [])
            if tasks:
                task = tasks[0]
                print(f"    {subset_name}:")
                print(f"      Original: {Path(task['original_path']).name}")
                print(f"      Synthetic: {Path(task['synthetic_path']).name}")
                print(f"      Dimensions: {task['width']}×{task['height']}")
                print(f"      Caption: {task['caption'][:50]}...")
                break
    
    return synthetic_path, generation_plan


def test_generation_placeholder(train_dir: str, output_dir: str, percent: float = 100.0):
    """Test image generation with placeholder images"""
    from library.neon_train_utils import generate_synthetic_dataset
    
    print("\n" + "=" * 60)
    print("PHASE 2: Synthetic Image Generation (Test Mode)")
    print("=" * 60)
    
    args = Args(output_dir, seed=42)
    args.neon_synthetic_dataset_dir = output_dir
    
    synthetic_path, generation_plan = generate_synthetic_dataset(
        args=args,
        train_data_dir=train_dir,
        synthetic_percent=percent,
        test_mode=True,  # Use placeholder images
    )
    
    print("\n✓ Test generation completed")
    
    # Verify outputs
    synthetic_dir = Path(synthetic_path)
    synthetic_images = []
    for ext in ['.png', '.jpg', '.jpeg', '.webp']:
        synthetic_images.extend(list(synthetic_dir.rglob(f'*_synthetic{ext}')))
    
    print(f"\n✓ Generated {len(synthetic_images)} synthetic images")
    
    # Check captions
    captions_found = 0
    for img_path in synthetic_images[:5]:  # Check first 5
        caption_path = img_path.with_suffix('.txt')
        if caption_path.exists():
            captions_found += 1
    
    print(f"✓ Found {captions_found}/{min(5, len(synthetic_images))} captions (checked)")
    
    # Show sample
    if synthetic_images:
        sample = synthetic_images[0]
        print(f"\nSample output:")
        print(f"  Image: {sample}")
        caption_file = sample.with_suffix('.txt')
        if caption_file.exists():
            with open(caption_file, 'r') as f:
                print(f"  Caption: {f.read()[:100]}...")
    
    return synthetic_path


def verify_incremental_generation(train_dir: str, output_dir: str):
    """Test that incremental generation works (reuses existing)"""
    from library.neon_train_utils import generate_synthetic_dataset, count_synthetic_images
    
    print("\n" + "=" * 60)
    print("PHASE 3: Incremental Generation Test")
    print("=" * 60)
    
    args = Args(output_dir, seed=42)
    args.neon_synthetic_dataset_dir = output_dir
    
    # Count existing before
    synthetic_dir = Path(output_dir)
    existing_before = count_synthetic_images(synthetic_dir)
    print(f"\nExisting synthetic images: {existing_before}")
    
    # Run generation again (should skip existing)
    print(f"Running generation again (should reuse existing)...")
    synthetic_path, generation_plan = generate_synthetic_dataset(
        args=args,
        train_data_dir=train_dir,
        synthetic_percent=100.0,
        test_mode=True,
    )
    
    # Count after
    existing_after = count_synthetic_images(synthetic_dir)
    print(f"Synthetic images after: {existing_after}")
    
    if existing_after == existing_before:
        print(f"✓ Incremental generation works! No duplicate generation.")
    else:
        print(f"⚠️  Generated {existing_after - existing_before} new images (unexpected)")
    
    # Test with higher percentage
    print(f"\nTesting with 150% (should generate more)...")
    synthetic_path, generation_plan = generate_synthetic_dataset(
        args=args,
        train_data_dir=train_dir,
        synthetic_percent=150.0,
        test_mode=True,
    )
    
    existing_final = count_synthetic_images(synthetic_dir)
    print(f"Synthetic images with 150%: {existing_final}")
    
    if existing_final > existing_after:
        print(f"✓ Percentage scaling works! Generated {existing_final - existing_after} additional images.")


def main():
    parser = argparse.ArgumentParser(
        description="Test Neon synthetic image generation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--train_dir", type=str, required=True, help="Original training data directory")
    parser.add_argument("--output", type=str, required=True, help="Output directory for synthetic data")
    parser.add_argument("--percent", type=float, default=100.0, help="Synthetic image percentage")
    parser.add_argument("--test_mode", action="store_true", help="Use placeholder images (no model loading)")
    parser.add_argument("--skip_incremental", action="store_true", help="Skip incremental generation test")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Neon Synthetic Image Generation Pipeline Test")
    print("=" * 60)
    print(f"\nTrain directory: {args.train_dir}")
    print(f"Output directory: {args.output}")
    print(f"Percentage: {args.percent}%")
    print(f"Test mode: {args.test_mode}")
    
    # Verify train directory exists
    if not Path(args.train_dir).exists():
        print(f"\n❌ Error: Training directory not found: {args.train_dir}")
        return 1
    
    # Test 1: Structure and plan
    try:
        synthetic_path, generation_plan = test_structure_and_plan(
            args.train_dir,
            args.output,
            args.percent
        )
    except Exception as e:
        print(f"\n❌ Phase 1 failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Test 2: Image generation (placeholder mode)
    if args.test_mode:
        try:
            test_generation_placeholder(args.train_dir, args.output, args.percent)
        except Exception as e:
            print(f"\n❌ Phase 2 failed: {e}")
            import traceback
            traceback.print_exc()
            return 1
        
        # Test 3: Incremental generation
        if not args.skip_incremental:
            try:
                verify_incremental_generation(args.train_dir, args.output)
            except Exception as e:
                print(f"\n❌ Phase 3 failed: {e}")
                import traceback
                traceback.print_exc()
                return 1
    else:
        print("\n⚠️  Skipping image generation (test_mode not enabled)")
        print("    Enable --test_mode to generate placeholder images")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed successfully!")
    print("=" * 60)
    print(f"\nSynthetic dataset created at: {args.output}")
    print("\nNext steps:")
    print("  1. Review generated synthetic images")
    print("  2. Check caption files match image names")
    print("  3. Test with actual model loading (remove --test_mode)")
    print("  4. Integrate into full Neon training workflow")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
