"""
Dataset Verification Script
Checks if CamVid dataset is properly installed

Usage:
    python scripts/verify_dataset.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
import config


def verify_dataset():
    """Verify CamVid dataset structure and integrity"""
    print("\n" + "="*70)
    print("VERIFYING CamVid DATASET")
    print("="*70 + "\n")
    
    camvid_dir = config.RAW_DATA_DIR / "camvid"
    
    if not camvid_dir.exists():
        print("✗ Dataset directory not found!")
        print(f"Expected location: {camvid_dir}")
        print("\nPlease download the dataset first.")
        print("See DATASET_SETUP.md for instructions.")
        return False
    
    print(f"Dataset location: {camvid_dir}\n")
    
    # Check all required folders
    splits = ['train', 'val', 'test']
    total_images = 0
    total_labels = 0
    all_ok = True
    
    for split in splits:
        img_dir = camvid_dir / split
        annot_dir = camvid_dir / f"{split}annot"
        
        # Check image directory
        if not img_dir.exists():
            print(f"✗ Missing folder: {split}/")
            all_ok = False
            continue
        
        # Check annotation directory
        if not annot_dir.exists():
            print(f"✗ Missing folder: {split}annot/")
            all_ok = False
            continue
        
        # Count files
        images = list(img_dir.glob("*.png"))
        labels = list(annot_dir.glob("*.png"))
        
        num_images = len(images)
        num_labels = len(labels)
        
        total_images += num_images
        total_labels += num_labels
        
        # Expected counts
        expected = {'train': 367, 'val': 101, 'test': 233}
        
        if num_images == expected[split] and num_labels == expected[split]:
            print(f"✓ {split:5s}: {num_images:3d} images, {num_labels:3d} labels")
        else:
            print(f"⚠ {split:5s}: {num_images:3d} images, {num_labels:3d} labels (expected {expected[split]})")
            if num_images == 0 or num_labels == 0:
                all_ok = False
        
        # Check if counts match
        if num_images != num_labels:
            print(f"  ⚠ Warning: Number of images and labels don't match for {split}")
    
    print(f"\n{'='*70}")
    
    if all_ok and total_images > 0:
        print("✓ DATASET VERIFICATION PASSED!")
        print(f"✓ Total: {total_images} images, {total_labels} labels")
        print(f"{'='*70}\n")
        print("Next steps:")
        print("1. Visualize samples: python scripts/visualize_dataset.py")
        print("2. Start training: python scripts/train.py")
        print("3. Run demo: python scripts/demo_navigation.py")
        return True
    else:
        print("✗ DATASET VERIFICATION FAILED!")
        print(f"{'='*70}\n")
        print("Please check:")
        print("1. All 6 folders exist (train, trainannot, val, valannot, test, testannot)")
        print("2. Each folder contains .png files")
        print("3. You extracted the ZIP file to the correct location")
        print("\nSee DATASET_SETUP.md for detailed instructions.")
        return False


def show_sample_info():
    """Show information about a random sample"""
    import numpy as np
    from PIL import Image
    
    camvid_dir = config.RAW_DATA_DIR / "camvid"
    train_dir = camvid_dir / "train"
    
    if not train_dir.exists():
        return
    
    # Get a random image
    images = list(train_dir.glob("*.png"))
    if not images:
        return
    
    import random
    sample_img = random.choice(images)
    sample_label = camvid_dir / "trainannot" / sample_img.name
    
    print(f"\nSample Image Info:")
    print(f"  File: {sample_img.name}")
    
    # Load image
    img = Image.open(sample_img)
    print(f"  Size: {img.size[0]}x{img.size[1]} pixels")
    print(f"  Mode: {img.mode}")
    
    if sample_label.exists():
        label = Image.open(sample_label)
        label_array = np.array(label)
        unique_classes = np.unique(label_array)
        print(f"  Classes in this image: {len(unique_classes)}")
        print(f"  Class IDs: {unique_classes.tolist()}")


if __name__ == "__main__":
    success = verify_dataset()
    
    if success:
        try:
            show_sample_info()
        except Exception as e:
            pass  # Don't fail if we can't show sample info
    
    sys.exit(0 if success else 1)
