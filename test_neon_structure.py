#!/usr/bin/env python3
"""
Test script for Neon dataset structure replication and incremental generation

Usage:
    # First run - creates structure
    python test_neon_structure.py --train_dir ./my_dataset --output ./test_synthetic

    # Second run - reuses existing, generates delta
    python test_neon_structure.py --train_dir ./my_dataset --output ./test_synthetic
"""

import sys
from pathlib import Path

# Add library to path
sys.path.insert(0, str(Path(__file__).parent / "library"))

from library.neon_train_utils import replicate_dataset_structure, is_synthetic_image, count_synthetic_images


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test Neon dataset structure replication with synthetic image detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Create new synthetic structure
    python test_neon_structure.py --train_dir ./my_data --output ./synthetic_data

    # Reuse existing (only generates missing synthetic images)
    python test_neon_structure.py --train_dir ./my_data --output ./synthetic_data --percent 150

    # Test synthetic image detection
    python test_neon_structure.py --train_dir ./my_data --output ./synthetic_data --test-detection

Note: Synthetic images are detected by '_synthetic' suffix in filename
      Example: img001_synthetic.png, img002_synthetic.jpg
        """
    )
    parser.add_argument("--train_dir", type=str, required=True, help="Original training data directory")
    parser.add_argument("--output", type=str, required=True, help="Output directory for synthetic structure")
    parser.add_argument("--percent", type=float, default=100.0, help="Synthetic image percentage (default: 100)")
    parser.add_argument("--test-detection", action="store_true", help="Test synthetic image detection")
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("Testing Neon Dataset Structure Replication")
    print("=" * 60)
    
    # Test synthetic image detection
    if args.test_detection:
        print(f"\nSynthetic Image Detection Test:")
        test_files = [
            "img001_synthetic.png",
            "img002_synthetic.jpg",
            "img003.png",
            "myimage_synthetic.webp",
            "regular_image.jpg",
        ]
        for filename in test_files:
            is_synth = is_synthetic_image(filename)
            status = "✓ SYNTHETIC" if is_synth else "✗ regular"
            print(f"  {filename:30s} → {status}")
        print()
    
    # Test structure replication
    synthetic_path, generation_plan = replicate_dataset_structure(
        train_data_dir=args.train_dir,
        output_dir=args.output,
        synthetic_percent=args.percent,
        reuse_existing=True,
    )
    
    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)
    print(f"\nSynthetic dataset at: {synthetic_path}")
    
    # Show generation plan summary
    total_target = sum(p['target'] for p in generation_plan.values())
    total_existing = sum(p['existing'] for p in generation_plan.values())
    total_to_generate = sum(p['to_generate'] for p in generation_plan.values())
    
    print(f"\nGeneration Summary:")
    print(f"  Total target:     {total_target} images")
    print(f"  Already exists:   {total_existing} images")
    print(f"  Will generate:    {total_to_generate} images")
    
    if total_existing > 0:
        print(f"\n✓ Incremental generation: Reusing {total_existing} existing images!")
    
    print("\nPer-subset breakdown:")
    for subset_name, info in generation_plan.items():
        status = "✓ reusing" if info['existing'] > 0 else "new"
        print(f"  {subset_name}:")
        print(f"    Target: {info['target']}, Existing: {info['existing']}, Generate: {info['to_generate']} ({status})")
    
    print("\nNext steps:")
    print("  1. Run this command again to test incremental generation")
    print("  2. Manually add some images to synthetic dirs to test delta detection")
    print("  3. Implement actual image generation using this plan")
    print()


if __name__ == "__main__":
    main()
