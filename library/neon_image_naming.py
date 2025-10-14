"""
Helper utilities for Neon synthetic image naming conventions

Synthetic images are identified by the '_synthetic' suffix in their filename:
- img001_synthetic.png
- character_pose_001_synthetic.jpg
- bg_landscape_synthetic.webp
"""

from pathlib import Path
from typing import List


def make_synthetic_filename(original_path: str, index: int = None) -> str:
    """
    Create a synthetic image filename from an original image path.
    
    Args:
        original_path: Path to original image (e.g., "img001.png")
        index: Optional index for synthetic image (e.g., 1 for "img001_synthetic_1.png")
    
    Returns:
        Synthetic filename with _synthetic suffix
    
    Examples:
        >>> make_synthetic_filename("img001.png")
        "img001_synthetic.png"
        
        >>> make_synthetic_filename("img001.png", 2)
        "img001_synthetic_2.png"
        
        >>> make_synthetic_filename("character_pose.jpg")
        "character_pose_synthetic.jpg"
    """
    path = Path(original_path)
    stem = path.stem  # filename without extension
    ext = path.suffix  # .png, .jpg, etc.
    
    if index is not None:
        return f"{stem}_synthetic_{index}{ext}"
    else:
        return f"{stem}_synthetic{ext}"


def is_synthetic_filename(filename: str) -> bool:
    """
    Check if a filename represents a synthetic image.
    
    Args:
        filename: Filename to check
    
    Returns:
        True if filename contains _synthetic before extension
    
    Examples:
        >>> is_synthetic_filename("img001_synthetic.png")
        True
        
        >>> is_synthetic_filename("img001.png")
        False
        
        >>> is_synthetic_filename("img_synthetic_2.jpg")
        True
    """
    path = Path(filename)
    return '_synthetic' in path.stem


def get_synthetic_images(directory: Path, pattern: str = "*") -> List[Path]:
    """
    Get all synthetic images in a directory.
    
    Args:
        directory: Directory to search
        pattern: Glob pattern to match (default: all images)
    
    Returns:
        List of paths to synthetic images
    
    Example:
        >>> synthetic_imgs = get_synthetic_images(Path("./output"), "*.png")
        >>> len(synthetic_imgs)
        42
    """
    if not directory.exists():
        return []
    
    image_extensions = ['.png', '.jpg', '.jpeg', '.webp', '.bmp']
    synthetic_images = []
    
    for img_path in directory.glob(pattern):
        if img_path.suffix.lower() in image_extensions and is_synthetic_filename(img_path.name):
            synthetic_images.append(img_path)
    
    return synthetic_images


def get_next_synthetic_index(directory: Path, base_stem: str, extension: str = ".png") -> int:
    """
    Get the next available synthetic image index for a base filename.
    
    Args:
        directory: Directory containing images
        base_stem: Base filename stem (without extension)
        extension: Image extension (default: .png)
    
    Returns:
        Next available index (1-based)
    
    Example:
        If directory contains:
        - img001_synthetic.png
        - img001_synthetic_1.png
        - img001_synthetic_2.png
        
        >>> get_next_synthetic_index(Path("./output"), "img001")
        3
    """
    if not directory.exists():
        return 0
    
    # Find all synthetic images with this base stem
    pattern = f"{base_stem}_synthetic*{extension}"
    existing = list(directory.glob(pattern))
    
    if not existing:
        return 0  # No synthetic images yet, start with base name
    
    # Extract indices
    indices = []
    for path in existing:
        stem = path.stem
        # Check for indexed synthetic: img001_synthetic_2
        if '_synthetic_' in stem:
            try:
                idx = int(stem.split('_synthetic_')[-1])
                indices.append(idx)
            except ValueError:
                pass
        # Base synthetic: img001_synthetic
        elif stem.endswith('_synthetic'):
            indices.append(0)
    
    if not indices:
        return 0
    
    return max(indices) + 1


def generate_synthetic_filename_sequence(
    base_name: str,
    count: int,
    extension: str = ".png",
    start_index: int = 0
) -> List[str]:
    """
    Generate a sequence of synthetic filenames.
    
    Args:
        base_name: Base filename (without extension or _synthetic)
        count: Number of filenames to generate
        extension: File extension (default: .png)
        start_index: Starting index (default: 0 for base _synthetic name)
    
    Returns:
        List of synthetic filenames
    
    Example:
        >>> generate_synthetic_filename_sequence("img", 3, ".png", 0)
        ['img_synthetic.png', 'img_synthetic_1.png', 'img_synthetic_2.png']
    """
    filenames = []
    for i in range(count):
        idx = start_index + i
        if idx == 0:
            filenames.append(f"{base_name}_synthetic{extension}")
        else:
            filenames.append(f"{base_name}_synthetic_{idx}{extension}")
    return filenames


if __name__ == "__main__":
    # Test the naming utilities
    print("Testing Neon Synthetic Image Naming Utilities\n")
    
    print("1. Creating synthetic filenames:")
    print(f"   {make_synthetic_filename('img001.png')}")
    print(f"   {make_synthetic_filename('img001.png', 2)}")
    print(f"   {make_synthetic_filename('character_pose.jpg')}")
    
    print("\n2. Detecting synthetic images:")
    test_names = [
        "img001_synthetic.png",
        "img001.png",
        "img_synthetic_5.jpg",
        "regular.jpg",
    ]
    for name in test_names:
        is_synth = is_synthetic_filename(name)
        print(f"   {name:30s} → {'SYNTHETIC' if is_synth else 'regular'}")
    
    print("\n3. Generating filename sequence:")
    sequence = generate_synthetic_filename_sequence("img", 5, ".png", 0)
    for filename in sequence:
        print(f"   {filename}")
