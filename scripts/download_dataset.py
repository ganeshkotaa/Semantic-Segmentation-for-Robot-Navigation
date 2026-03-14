"""
Dataset Download Script for CamVid Dataset
Downloads and prepares the CamVid dataset for semantic segmentation

CamVid Dataset:
- 367 training images
- 101 validation images  
- 233 test images
- 12 semantic classes
- 360x480 resolution
- Perfect for Colab free tier (~700MB total)

Usage:
    python scripts/download_dataset.py
"""

import os
import sys
import zipfile
import urllib.request
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
import config


class DownloadProgressBar(tqdm):
    """Progress bar for urllib downloads"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    """Download file from URL with progress bar"""
    try:
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
            urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
        return True
    except Exception as e:
        print(f"✗ Download failed: {e}")
        return False


def extract_zip(zip_path, extract_to):
    """Extract ZIP file"""
    print(f"Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"✓ Extracted to {extract_to}")


def download_camvid():
    """Download and extract CamVid dataset"""
    print("="*70)
    print("DOWNLOADING CamVid DATASET")
    print("="*70)
    print("\nDataset Info:")
    print("- Source: Cambridge-driving Labeled Video Database")
    print("- Images: 701 total (367 train, 101 val, 233 test)")
    print("- Resolution: 360x480 pixels")
    print("- Classes: 12 semantic classes")
    print("- Size: ~700MB")
    print("- Best for: Quick prototyping and Colab free tier")
    print("="*70)
    
    # Create directories
    camvid_dir = config.RAW_DATA_DIR / "camvid"
    camvid_dir.mkdir(parents=True, exist_ok=True)
    
    temp_dir = camvid_dir / "temp"
    temp_dir.mkdir(exist_ok=True)
    
    # Download all files
    print("\n[1/3] Downloading dataset files...")
    failed_downloads = []
    
    for name, url in config.CAMVID_URLS.items():
        output_file = temp_dir / f"{name}.zip"
        
        # Skip if already downloaded
        if output_file.exists():
            print(f"✓ {name}.zip already exists, skipping...")
            continue
        
        print(f"\nDownloading {name}.zip...")
        success = download_url(url, output_file)
        
        if not success:
            failed_downloads.append(name)
            print(f"✗ Failed to download {name}.zip")
    
    # Handle failed downloads
    if failed_downloads:
        print("\n" + "="*70)
        print("⚠ AUTOMATIC DOWNLOAD FAILED")
        print("="*70)
        print("\nSome files could not be downloaded automatically.")
        print("Please download CamVid manually using one of these options:\n")
        print("OPTION 1 - Kaggle (Recommended):")
        print("1. Visit: https://www.kaggle.com/datasets/carlolepelaars/camvid")
        print("2. Click 'Download' (requires free Kaggle account)")
        print("3. Extract to:", camvid_dir)
        print("\nOPTION 2 - Google Drive:")
        print("1. Visit: https://drive.google.com/drive/folders/0B0ZXjo_p8lHBfms3X0JJbGFYQ2M1WWxNWEpJN0VfY3B4YlJuaGFnQUJvNDBBMWM1NVJxeWc")
        print("2. Download 'CamVid.zip'")
        print("3. Extract to:", camvid_dir)
        print("\nOPTION 3 - Alternative Mirrors:")
        print("- GitHub: https://github.com/mostafaizz/camvid")
        print("- Alternative: http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/")
        print("\nExpected folder structure after extraction:")
        print(f"{camvid_dir}/")
        print("  ├── train/        (367 images)")
        print("  ├── trainannot/   (367 labels)")
        print("  ├── val/          (101 images)")
        print("  ├── valannot/     (101 labels)")
        print("  ├── test/         (233 images)")
        print("  └── testannot/    (233 labels)")
        print("="*70)
        return False
    
    # Extract all files
    print("\n[2/3] Extracting files...")
    for zip_file in temp_dir.glob("*.zip"):
        extract_zip(zip_file, camvid_dir)
    
    # Organize directory structure
    print("\n[3/3] Organizing directory structure...")
    
    # Expected structure after extraction:
    # camvid/train/, camvid/val/, camvid/test/
    # camvid/trainannot/, camvid/valannot/, camvid/testannot/
    
    splits = ['train', 'val', 'test']
    for split in splits:
        img_dir = camvid_dir / split
        annot_dir = camvid_dir / f"{split}annot"
        
        if img_dir.exists() and annot_dir.exists():
            num_images = len(list(img_dir.glob("*.png")))
            num_labels = len(list(annot_dir.glob("*.png")))
            print(f"✓ {split}: {num_images} images, {num_labels} labels")
        else:
            print(f"✗ {split} directory not found!")
    
    # Clean up temp directory
    print("\n[4/3] Cleaning up...")
    for zip_file in temp_dir.glob("*.zip"):
        zip_file.unlink()
    temp_dir.rmdir()
    print("✓ Temporary files removed")
    
    print("\n" + "="*70)
    print("✓ DATASET DOWNLOAD COMPLETE!")
    print("="*70)
    print(f"\nDataset location: {camvid_dir}")
    print("\nNext steps:")
    print("1. Run visualization script: python scripts/visualize_dataset.py")
    print("2. Start training: python scripts/train.py")
    print("="*70)
    
    return True


def verify_dataset():
    """Verify dataset integrity"""
    print("\n" + "="*70)
    print("VERIFYING DATASET")
    print("="*70)
    
    camvid_dir = config.RAW_DATA_DIR / "camvid"
    
    if not camvid_dir.exists():
        print("✗ Dataset directory not found!")
        return False
    
    splits = ['train', 'val', 'test']
    expected_counts = {'train': 367, 'val': 101, 'test': 233}
    
    all_good = True
    for split in splits:
        img_dir = camvid_dir / split
        annot_dir = camvid_dir / f"{split}annot"
        
        if not img_dir.exists() or not annot_dir.exists():
            print(f"✗ {split} directories missing!")
            all_good = False
            continue
        
        num_images = len(list(img_dir.glob("*.png")))
        num_labels = len(list(annot_dir.glob("*.png")))
        
        if num_images == expected_counts[split] and num_images == num_labels:
            print(f"✓ {split}: {num_images} images, {num_labels} labels (OK)")
        else:
            print(f"✗ {split}: {num_images} images, {num_labels} labels (expected {expected_counts[split]})")
            all_good = False
    
    print("="*70)
    if all_good:
        print("✓ DATASET VERIFICATION PASSED")
    else:
        print("✗ DATASET VERIFICATION FAILED")
    print("="*70)
    
    return all_good


def main():
    """Main function"""
    print("\n" + "="*70)
    print("CamVid Dataset Downloader")
    print("Semantic Segmentation for Robot Navigation")
    print("="*70)
    
    # Check if dataset already exists
    camvid_dir = config.RAW_DATA_DIR / "camvid"
    if (camvid_dir / "train").exists():
        print("\n⚠ Dataset already exists!")
        response = input("Do you want to re-download? (y/n): ")
        if response.lower() != 'y':
            print("Skipping download. Verifying existing dataset...")
            verify_dataset()
            return
    
    # Download dataset
    success = download_camvid()
    
    if success:
        # Verify
        verify_dataset()
    else:
        print("\n✗ Download failed. Please check your internet connection and try again.")


if __name__ == "__main__":
    main()
