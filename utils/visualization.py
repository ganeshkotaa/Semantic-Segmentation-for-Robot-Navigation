"""
Visualization utilities for semantic segmentation
Includes visualization of predictions, overlays, cost maps, and path planning
"""

import sys
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import torch
import cv2

sys.path.append(str(Path(__file__).parent.parent))
import config


def denormalize_image(image: torch.Tensor, mean: List[float] = None, std: List[float] = None) -> np.ndarray:
    """
    Denormalize image tensor for visualization
    
    Args:
        image: Normalized image tensor [C, H, W]
        mean: Mean used for normalization
        std: Std used for normalization
    
    Returns:
        Denormalized image as numpy array [H, W, C]
    """
    if mean is None:
        mean = config.IMAGENET_MEAN
    if std is None:
        std = config.IMAGENET_STD
    
    # Clone to avoid modifying original
    image = image.clone()
    
    # Denormalize
    for c in range(3):
        image[c] = image[c] * std[c] + mean[c]
    
    # Clip to [0, 1]
    image = torch.clamp(image, 0, 1)
    
    # Convert to numpy [H, W, C]
    image = image.permute(1, 2, 0).cpu().numpy()
    
    # Convert to uint8
    image = (image * 255).astype(np.uint8)
    
    return image


def create_color_map(num_classes: int = 12) -> np.ndarray:
    """
    Create color map for visualization
    
    Args:
        num_classes: Number of classes
    
    Returns:
        Color map array [num_classes, 3] with RGB values
    """
    if num_classes <= len(config.CLASS_COLORS):
        return np.array(config.CLASS_COLORS[:num_classes], dtype=np.uint8)
    
    # Generate random colors if not enough predefined
    colors = config.CLASS_COLORS.copy()
    while len(colors) < num_classes:
        colors.append([np.random.randint(0, 255) for _ in range(3)])
    
    return np.array(colors[:num_classes], dtype=np.uint8)


def label_to_color(label: np.ndarray, color_map: np.ndarray) -> np.ndarray:
    """
    Convert label mask to RGB color image
    
    Args:
        label: Label mask [H, W] with class indices
        color_map: Color map [num_classes, 3]
    
    Returns:
        RGB image [H, W, 3]
    """
    h, w = label.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_idx in range(len(color_map)):
        mask = label == class_idx
        colored[mask] = color_map[class_idx]
    
    return colored


def visualize_prediction(
    image: torch.Tensor,
    prediction: torch.Tensor,
    target: Optional[torch.Tensor] = None,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Visualize segmentation prediction with optional ground truth
    
    Args:
        image: Input image tensor [C, H, W] (normalized)
        prediction: Prediction tensor [C, H, W] or [H, W]
        target: Optional ground truth tensor [H, W]
        class_names: Optional list of class names
        save_path: Optional path to save figure
        show: Whether to display the figure
    """
    # Denormalize image
    img_np = denormalize_image(image)
    
    # Convert prediction to class indices
    if prediction.dim() == 3:  # [C, H, W]
        pred_np = torch.argmax(prediction, dim=0).cpu().numpy()
    else:  # [H, W]
        pred_np = prediction.cpu().numpy()
    
    # Get color map
    num_classes = len(class_names) if class_names else config.NUM_CLASSES
    color_map = create_color_map(num_classes)
    
    # Convert to colored images
    pred_colored = label_to_color(pred_np, color_map)
    
    # Create overlay
    overlay = cv2.addWeighted(img_np, 0.6, pred_colored, 0.4, 0)
    
    # Setup plot
    if target is not None:
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        target_np = target.cpu().numpy()
        target_colored = label_to_color(target_np, color_map)
        
        axes[0].imshow(img_np)
        axes[0].set_title('Input Image')
        axes[0].axis('off')
        
        axes[1].imshow(target_colored)
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        axes[2].imshow(pred_colored)
        axes[2].set_title('Prediction')
        axes[2].axis('off')
        
        axes[3].imshow(overlay)
        axes[3].set_title('Overlay')
        axes[3].axis('off')
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(img_np)
        axes[0].set_title('Input Image')
        axes[0].axis('off')
        
        axes[1].imshow(pred_colored)
        axes[1].set_title('Prediction')
        axes[1].axis('off')
        
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
    
    # Add legend
    if class_names:
        patches = [mpatches.Patch(color=np.array(color_map[i])/255.0, label=class_names[i]) 
                   for i in range(min(len(class_names), len(color_map)))]
        fig.legend(handles=patches, loc='lower center', ncol=6, frameon=False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved visualization to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def visualize_batch(
    images: torch.Tensor,
    predictions: torch.Tensor,
    targets: Optional[torch.Tensor] = None,
    max_images: int = 4,
    save_path: Optional[str] = None
):
    """
    Visualize a batch of predictions
    
    Args:
        images: Batch of images [B, C, H, W]
        predictions: Batch of predictions [B, C, H, W] or [B, H, W]
        targets: Optional batch of ground truth [B, H, W]
        max_images: Maximum number of images to show
        save_path: Optional path to save figure
    """
    batch_size = min(images.shape[0], max_images)
    
    # Get color map
    color_map = create_color_map()
    
    # Setup plot
    if targets is not None:
        fig, axes = plt.subplots(batch_size, 4, figsize=(16, 4*batch_size))
        if batch_size == 1:
            axes = axes.reshape(1, -1)
    else:
        fig, axes = plt.subplots(batch_size, 3, figsize=(12, 4*batch_size))
        if batch_size == 1:
            axes = axes.reshape(1, -1)
    
    for i in range(batch_size):
        # Denormalize image
        img_np = denormalize_image(images[i])
        
        # Convert prediction
        if predictions.dim() == 4:  # [B, C, H, W]
            pred_np = torch.argmax(predictions[i], dim=0).cpu().numpy()
        else:  # [B, H, W]
            pred_np = predictions[i].cpu().numpy()
        pred_colored = label_to_color(pred_np, color_map)
        
        # Create overlay
        overlay = cv2.addWeighted(img_np, 0.6, pred_colored, 0.4, 0)
        
        # Plot
        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title(f'Image {i+1}')
        axes[i, 0].axis('off')
        
        if targets is not None:
            target_np = targets[i].cpu().numpy()
            target_colored = label_to_color(target_np, color_map)
            
            axes[i, 1].imshow(target_colored)
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(pred_colored)
            axes[i, 2].set_title('Prediction')
            axes[i, 2].axis('off')
            
            axes[i, 3].imshow(overlay)
            axes[i, 3].set_title('Overlay')
            axes[i, 3].axis('off')
        else:
            axes[i, 1].imshow(pred_colored)
            axes[i, 1].set_title('Prediction')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(overlay)
            axes[i, 2].set_title('Overlay')
            axes[i, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved batch visualization to {save_path}")
    
    plt.show()


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_ious: List[float],
    val_ious: List[float],
    save_path: Optional[str] = None
):
    """
    Plot training and validation curves
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        train_ious: List of training IoUs
        val_ious: List of validation IoUs
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curve
    epochs = range(1, len(train_losses) + 1)
    axes[0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # IoU curve
    axes[1].plot(epochs, train_ious, 'b-', label='Train mIoU', linewidth=2)
    axes[1].plot(epochs, val_ious, 'r-', label='Val mIoU', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('mIoU')
    axes[1].set_title('Training and Validation mIoU')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved training curves to {save_path}")
    
    plt.show()


def save_prediction_comparison(
    image: np.ndarray,
    prediction: np.ndarray,
    target: np.ndarray,
    save_path: str,
    class_names: Optional[List[str]] = None
):
    """
    Save side-by-side comparison of image, ground truth, and prediction
    
    Args:
        image: Input image [H, W, 3]
        prediction: Prediction mask [H, W]
        target: Ground truth mask [H, W]
        save_path: Path to save image
        class_names: Optional list of class names
    """
    color_map = create_color_map()
    
    pred_colored = label_to_color(prediction, color_map)
    target_colored = label_to_color(target, color_map)
    
    # Create side-by-side image
    combined = np.hstack([image, target_colored, pred_colored])
    
    # Save
    Image.fromarray(combined).save(save_path)


# ═══════════════════════════════════════════════════════════════════
# INIT FILES
# ═══════════════════════════════════════════════════════════════════

def create_init_files():
    """Create __init__.py files for package structure"""
    utils_init = Path(__file__).parent / "__init__.py"
    models_init = Path(__file__).parent.parent / "models" / "__init__.py"
    
    # Create empty __init__.py files
    for init_file in [utils_init, models_init]:
        if not init_file.exists():
            init_file.write_text("")
            print(f"✓ Created {init_file}")


if __name__ == "__main__":
    create_init_files()
    print("\n✓ Visualization utilities ready!")
