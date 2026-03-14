"""
PyTorch Dataset class for CamVid semantic segmentation
Handles data loading, preprocessing, and augmentation
"""

import os
import sys
from pathlib import Path
from typing import Tuple, Dict, Optional, Callable

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
import config


# CamVid RGB label colors to class index mapping
CAMVID_LABEL_COLORS = {
    (128, 128, 128): 0,   # Sky
    (128, 0, 0): 1,       # Building
    (192, 192, 128): 2,   # Pole
    (128, 64, 128): 3,    # Road
    (60, 40, 222): 4,     # Pavement  
    (128, 128, 0): 5,     # Tree
    (192, 128, 128): 6,   # SignSymbol
    (64, 64, 128): 7,     # Fence
    (64, 0, 128): 8,      # Car
    (64, 64, 0): 9,       # Pedestrian
    (0, 128, 192): 10,    # Bicyclist
    (0, 0, 0): 11         # Unlabelled/Void
}


def rgb_to_class_index(rgb_label: np.ndarray) -> np.ndarray:
    """
    Convert RGB-encoded CamVid label to class index label
    
    Args:
        rgb_label: RGB label image [H, W, 3] with dtype uint8
        
    Returns:
        class_label: Class index image [H, W] with dtype uint8
    """
    height, width = rgb_label.shape[:2]
    class_label = np.zeros((height, width), dtype=np.uint8)
    
    # Map each RGB color to its class index
    for rgb_color, class_idx in CAMVID_LABEL_COLORS.items():
        # Create mask where all 3 channels match the target RGB
        mask = np.all(rgb_label == rgb_color, axis=-1)
        class_label[mask] = class_idx
    
    return class_label


class CamVidDataset(Dataset):
    """
    CamVid Dataset for semantic segmentation
    
    Args:
        root_dir: Path to CamVid dataset root directory
        split: 'train', 'val', or 'test'
        transform: Albumentations transform pipeline
        img_size: Tuple of (height, width) for resizing
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        img_size: Tuple[int, int] = (360, 480)
    ):
        assert split in ['train', 'val', 'test'], f"Invalid split: {split}"
        
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.img_size = img_size
        
        # Set up paths
        self.images_dir = self.root_dir / split
        self.labels_dir = self.root_dir / f"{split}annot"
        
        # Verify directories exist
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not self.labels_dir.exists():
            raise FileNotFoundError(f"Labels directory not found: {self.labels_dir}")
        
        # Get list of image files
        self.image_files = sorted(list(self.images_dir.glob("*.png")))
        
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {self.images_dir}")
        
        # Build label file mapping
        self.label_files = []
        for img_file in self.image_files:
            # Label files have same name but in different directory
            label_file = self.labels_dir / img_file.name
            if not label_file.exists():
                # Try with _L suffix (some datasets use this)
                label_file = self.labels_dir / img_file.name.replace('.png', '_L.png')
            
            if label_file.exists():
                self.label_files.append(label_file)
            else:
                print(f"Warning: Label not found for {img_file.name}")
        
        print(f"Loaded {split} split: {len(self.image_files)} images, {len(self.label_files)} labels")
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset
        
        Returns:
            Dictionary with keys:
                - 'image': torch.Tensor [C, H, W]
                - 'label': torch.Tensor [H, W] with class indices
                - 'image_name': str
        """
        # Load image
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        # Load label
        label_path = self.label_files[idx] if idx < len(self.label_files) else None
        if label_path is not None:
            label = Image.open(label_path)
            label = np.array(label)
            
            # Convert RGB-encoded labels to class indices
            if len(label.shape) == 3 and label.shape[2] == 3:
                # CamVid labels are RGB-encoded - convert to class indices
                label = rgb_to_class_index(label)
            elif len(label.shape) == 3:
                # Has 3 channels but not RGB - take first channel
                label = label[:, :, 0]
            
            # Ensure labels are in valid range [0, NUM_CLASSES-1]
            label = np.clip(label, 0, config.NUM_CLASSES - 1).astype(np.uint8)
        else:
            # Create dummy label if missing
            label = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        # Apply transformations
        if self.transform is not None:
            transformed = self.transform(image=image, mask=label)
            image = transformed['image']
            label = transformed['mask']
            
            # Ensure label is Long tensor for CrossEntropyLoss
            if isinstance(label, torch.Tensor):
                label = label.long()
            else:
                label = torch.from_numpy(label).long()
        else:
            # Default: just convert to tensor
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            label = torch.from_numpy(label).long()
        
        return {
            'image': image,
            'label': label,
            'image_name': img_path.name
        }


def get_train_transform(img_size: Tuple[int, int] = (360, 480)) -> A.Compose:
    """
    Get training data augmentation pipeline
    
    Args:
        img_size: Target image size (height, width)
    
    Returns:
        Albumentations Compose object
    """
    return A.Compose([
        # Resize to target size
        A.Resize(height=img_size[0], width=img_size[1]),
        
        # Geometric augmentations
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, p=0.5),
        
        # Color augmentations
        A.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1,
            p=0.5
        ),
        
        # Normalize with ImageNet stats
        A.Normalize(
            mean=config.IMAGENET_MEAN,
            std=config.IMAGENET_STD
        ),
        
        # Convert to PyTorch tensor
        ToTensorV2()
    ])


def get_val_transform(img_size: Tuple[int, int] = (360, 480)) -> A.Compose:
    """
    Get validation data augmentation pipeline (minimal augmentation)
    
    Args:
        img_size: Target image size (height, width)
    
    Returns:
        Albumentations Compose object
    """
    return A.Compose([
        # Resize to target size
        A.Resize(height=img_size[0], width=img_size[1]),
        
        # Normalize with ImageNet stats
        A.Normalize(
            mean=config.IMAGENET_MEAN,
            std=config.IMAGENET_STD
        ),
        
        # Convert to PyTorch tensor
        ToTensorV2()
    ])


def create_dataloaders(
    data_dir: str,
    batch_size: int = 4,
    num_workers: int = 2,
    img_size: Tuple[int, int] = (360, 480)
) -> Dict[str, torch.utils.data.DataLoader]:
    """
    Create train, val, and test dataloaders
    
    Args:
        data_dir: Path to CamVid dataset root
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
        img_size: Target image size (height, width)
    
    Returns:
        Dictionary with 'train', 'val', 'test' dataloaders
    """
    # Create datasets
    train_dataset = CamVidDataset(
        root_dir=data_dir,
        split='train',
        transform=get_train_transform(img_size),
        img_size=img_size
    )
    
    val_dataset = CamVidDataset(
        root_dir=data_dir,
        split='val',
        transform=get_val_transform(img_size),
        img_size=img_size
    )
    
    test_dataset = CamVidDataset(
        root_dir=data_dir,
        split='test',
        transform=get_val_transform(img_size),
        img_size=img_size
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }


def test_dataset():
    """Test function to verify dataset loading"""
    print("="*70)
    print("Testing CamVid Dataset")
    print("="*70)
    
    # Create dataset
    data_dir = config.RAW_DATA_DIR / "camvid"
    
    try:
        dataset = CamVidDataset(
            root_dir=str(data_dir),
            split='train',
            transform=get_train_transform()
        )
        
        print(f"\n✓ Dataset created successfully")
        print(f"  - Number of samples: {len(dataset)}")
        
        # Get a sample
        sample = dataset[0]
        print(f"\n✓ Sample loaded successfully")
        print(f"  - Image shape: {sample['image'].shape}")
        print(f"  - Label shape: {sample['label'].shape}")
        print(f"  - Image dtype: {sample['image'].dtype}")
        print(f"  - Label dtype: {sample['label'].dtype}")
        print(f"  - Image range: [{sample['image'].min():.3f}, {sample['image'].max():.3f}]")
        print(f"  - Unique labels: {torch.unique(sample['label']).tolist()}")
        print(f"  - Image name: {sample['image_name']}")
        
        # Create dataloader
        dataloaders = create_dataloaders(
            data_dir=str(data_dir),
            batch_size=2,
            num_workers=0
        )
        
        print(f"\n✓ Dataloaders created successfully")
        print(f"  - Train batches: {len(dataloaders['train'])}")
        print(f"  - Val batches: {len(dataloaders['val'])}")
        print(f"  - Test batches: {len(dataloaders['test'])}")
        
        # Test batch loading
        batch = next(iter(dataloaders['train']))
        print(f"\n✓ Batch loaded successfully")
        print(f"  - Batch image shape: {batch['image'].shape}")
        print(f"  - Batch label shape: {batch['label'].shape}")
        
        print("\n" + "="*70)
        print("✓ ALL TESTS PASSED")
        print("="*70)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_dataset()
