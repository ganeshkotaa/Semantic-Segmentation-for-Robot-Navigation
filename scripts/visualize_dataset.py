"""
Dataset Visualization Script
Visualize CamVid dataset samples with labels

Usage:
    python scripts/visualize_dataset.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import config
from utils.dataset import CamVidDataset, get_train_transform, get_val_transform
from utils.visualization import create_color_map, label_to_color


def visualize_samples(split='train', num_samples=6, save_path=None):
    """
    Visualize random samples from dataset
    
    Args:
        split: Dataset split ('train', 'val', or 'test')
        num_samples: Number of samples to show
        save_path: Optional path to save figure
    """
    print(f"\n{'='*70}")
    print(f"VISUALIZING {split.upper()} DATASET")
    print(f"{'='*70}\n")
    
    # Load dataset
    data_dir = config.RAW_DATA_DIR / "camvid"
    
    dataset = CamVidDataset(
        root_dir=str(data_dir),
        split=split,
        transform=get_val_transform(),  # Use val transform for clean visualization
        img_size=config.INPUT_SIZE
    )
    
    print(f"Dataset: {len(dataset)} images")
    print(f"Classes: {len(config.CAMVID_CLASSES)}")
    print()
    
    # Get color map
    color_map = create_color_map()
    class_names = list(config.CAMVID_CLASSES.values())
    
    # Create figure
    fig = plt.figure(figsize=(20, 4 * num_samples))
    gs = GridSpec(num_samples, 3, figure=fig, hspace=0.3, wspace=0.2)
    
    # Sample indices
    indices = np.random.choice(len(dataset), size=min(num_samples, len(dataset)), replace=False)
    
    for i, idx in enumerate(indices):
        sample = dataset[idx]
        
        # Denormalize image
        image = sample['image'].permute(1, 2, 0).cpu().numpy()
        image = image * np.array(config.IMAGENET_STD) + np.array(config.IMAGENET_MEAN)
        image = np.clip(image, 0, 1)
        
        # Get label
        label = sample['label'].cpu().numpy()
        label_colored = label_to_color(label, color_map)
        
        # Create overlay
        import cv2
        image_uint8 = (image * 255).astype(np.uint8)
        overlay = cv2.addWeighted(image_uint8, 0.6, label_colored, 0.4, 0)
        
        # Plot image
        ax1 = fig.add_subplot(gs[i, 0])
        ax1.imshow(image)
        ax1.set_title(f"Image: {sample['image_name']}", fontsize=10)
        ax1.axis('off')
        
        # Plot label
        ax2 = fig.add_subplot(gs[i, 1])
        ax2.imshow(label_colored)
        ax2.set_title("Ground Truth", fontsize=10)
        ax2.axis('off')
        
        # Plot overlay
        ax3 = fig.add_subplot(gs[i, 2])
        ax3.imshow(overlay)
        ax3.set_title("Overlay", fontsize=10)
        ax3.axis('off')
        
        # Add class statistics for this image
        unique_classes = np.unique(label)
        class_text = ", ".join([class_names[c] for c in unique_classes[:5]])
        ax1.text(0.02, 0.98, f"Classes: {class_text}...", 
                transform=ax1.transAxes, fontsize=8,
                verticalalignment='top', bbox=dict(boxstyle='round', 
                facecolor='white', alpha=0.7))
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color_map[i]/255.0, label=class_names[i])
        for i in range(min(12, len(class_names)))
    ]
    
    fig.legend(handles=legend_elements, loc='upper center', ncol=6,
              bbox_to_anchor=(0.5, 0.02), frameon=True, fontsize=10)
    
    fig.suptitle(f'CamVid {split.capitalize()} Dataset Samples', 
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.99])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved visualization to {save_path}")
    
    plt.show()


def show_class_distribution(split='train'):
    """Show class distribution in dataset"""
    print(f"\nAnalyzing class distribution in {split} set...")
    
    data_dir = config.RAW_DATA_DIR / "camvid"
    dataset = CamVidDataset(
        root_dir=str(data_dir),
        split=split,
        transform=get_val_transform()
    )
    
    # Count pixels per class
    class_counts = np.zeros(config.NUM_CLASSES)
    
    for i in range(len(dataset)):
        sample = dataset[i]
        label = sample['label'].cpu().numpy()
        
        for class_idx in range(config.NUM_CLASSES):
            class_counts[class_idx] += (label == class_idx).sum()
    
    # Plot distribution
    fig, ax = plt.subplots(figsize=(12, 6))
    
    class_names = list(config.CAMVID_CLASSES.values())
    x = np.arange(len(class_names))
    
    bars = ax.bar(x, class_counts / class_counts.sum() * 100)
    
    # Color bars
    color_map = create_color_map()
    for bar, color in zip(bars, color_map):
        bar.set_color(color / 255.0)
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title(f'Class Distribution in {split.capitalize()} Set', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\nClass Distribution:")
    for i, name in enumerate(class_names):
        pct = class_counts[i] / class_counts.sum() * 100
        print(f"  {name:20s}: {pct:5.2f}%")


def main():
    """Main function"""
    print("\n" + "="*70)
    print("CamVid Dataset Visualization")
    print("="*70)
    
    # Check if dataset exists
    data_dir = config.RAW_DATA_DIR / "camvid"
    if not data_dir.exists():
        print("\n✗ Dataset not found!")
        print("Please run: python scripts/download_dataset.py")
        return
    
    # Visualize samples
    output_dir = config.RESULTS_DIR / "dataset_visualization"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for split in ['train', 'val', 'test']:
        save_path = output_dir / f"{split}_samples.png"
        visualize_samples(split=split, num_samples=4, save_path=str(save_path))
    
    # Show class distribution
    show_class_distribution('train')
    
    print("\n✓ Visualization complete!")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()
