"""
Evaluation metrics for semantic segmentation
Includes: mIoU, Pixel Accuracy, Dice Score, Per-Class IoU
"""

import torch
import numpy as np
from typing import Dict, List


class SegmentationMetrics:
    """
    Calculate segmentation metrics: mIoU, Pixel Accuracy, Dice Score
    """
    
    def __init__(self, num_classes: int, ignore_index: int = -1):
        """
        Args:
            num_classes: Number of segmentation classes
            ignore_index: Class index to ignore in calculations
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        Update metrics with new batch
        
        Args:
            predictions: Predicted class indices [B, H, W] or logits [B, C, H, W]
            targets: Ground truth class indices [B, H, W]
        """
        # Convert logits to class predictions if needed
        if predictions.dim() == 4:  # [B, C, H, W]
            predictions = torch.argmax(predictions, dim=1)  # [B, H, W]
        
        # Move to CPU and convert to numpy
        predictions = predictions.cpu().numpy().flatten()
        targets = targets.cpu().numpy().flatten()
        
        # Mask out ignore index
        mask = (targets >= 0) & (targets < self.num_classes)
        if self.ignore_index >= 0:
            mask = mask & (targets != self.ignore_index)
        
        predictions = predictions[mask]
        targets = targets[mask]
        
        # Update confusion matrix
        # confusion_matrix[i, j] = number of pixels with true class i predicted as class j
        for t, p in zip(targets, predictions):
            if 0 <= t < self.num_classes and 0 <= p < self.num_classes:
                self.confusion_matrix[t, p] += 1
    
    def get_miou(self) -> float:
        """
        Calculate mean Intersection over Union (mIoU)
        
        Returns:
            Mean IoU across all classes
        """
        # IoU = TP / (TP + FP + FN)
        # TP = diagonal, FP = column sum - TP, FN = row sum - TP
        
        iou_per_class = []
        
        for i in range(self.num_classes):
            tp = self.confusion_matrix[i, i]
            fp = self.confusion_matrix[:, i].sum() - tp
            fn = self.confusion_matrix[i, :].sum() - tp
            
            denominator = tp + fp + fn
            if denominator == 0:
                iou = float('nan')  # Ignore classes not present
            else:
                iou = tp / denominator
            
            iou_per_class.append(iou)
        
        # Calculate mean, ignoring NaN values
        iou_array = np.array(iou_per_class)
        valid_ious = iou_array[~np.isnan(iou_array)]
        
        if len(valid_ious) == 0:
            return 0.0
        
        return float(valid_ious.mean())
    
    def get_iou_per_class(self) -> Dict[int, float]:
        """
        Get IoU for each class
        
        Returns:
            Dictionary mapping class index to IoU
        """
        iou_dict = {}
        
        for i in range(self.num_classes):
            tp = self.confusion_matrix[i, i]
            fp = self.confusion_matrix[:, i].sum() - tp
            fn = self.confusion_matrix[i, :].sum() - tp
            
            denominator = tp + fp + fn
            if denominator == 0:
                iou = 0.0
            else:
                iou = tp / denominator
            
            iou_dict[i] = float(iou)
        
        return iou_dict
    
    def get_pixel_accuracy(self) -> float:
        """
        Calculate pixel accuracy
        
        Returns:
            Overall pixel accuracy
        """
        correct = np.diag(self.confusion_matrix).sum()
        total = self.confusion_matrix.sum()
        
        if total == 0:
            return 0.0
        
        return float(correct / total)
    
    def get_dice_score(self) -> float:
        """
        Calculate mean Dice score (F1 score)
        
        Returns:
            Mean Dice score across all classes
        """
        dice_per_class = []
        
        for i in range(self.num_classes):
            tp = self.confusion_matrix[i, i]
            fp = self.confusion_matrix[:, i].sum() - tp
            fn = self.confusion_matrix[i, :].sum() - tp
            
            denominator = 2 * tp + fp + fn
            if denominator == 0:
                dice = float('nan')
            else:
                dice = (2 * tp) / denominator
            
            dice_per_class.append(dice)
        
        # Calculate mean, ignoring NaN values
        dice_array = np.array(dice_per_class)
        valid_dice = dice_array[~np.isnan(dice_array)]
        
        if len(valid_dice) == 0:
            return 0.0
        
        return float(valid_dice.mean())
    
    def get_all_metrics(self) -> Dict[str, float]:
        """
        Get all metrics at once
        
        Returns:
            Dictionary with all metrics
        """
        return {
            'miou': self.get_miou(),
            'pixel_accuracy': self.get_pixel_accuracy(),
            'dice_score': self.get_dice_score()
        }
    
    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix"""
        return self.confusion_matrix
    
    def print_metrics(self, class_names: List[str] = None):
        """
        Print formatted metrics
        
        Args:
            class_names: Optional list of class names for display
        """
        print("\n" + "="*70)
        print("SEGMENTATION METRICS")
        print("="*70)
        
        # Overall metrics
        metrics = self.get_all_metrics()
        print(f"\nOverall Metrics:")
        print(f"  Mean IoU:        {metrics['miou']:.4f} ({metrics['miou']*100:.2f}%)")
        print(f"  Pixel Accuracy:  {metrics['pixel_accuracy']:.4f} ({metrics['pixel_accuracy']*100:.2f}%)")
        print(f"  Dice Score:      {metrics['dice_score']:.4f} ({metrics['dice_score']*100:.2f}%)")
        
        # Per-class IoU
        iou_per_class = self.get_iou_per_class()
        print(f"\nPer-Class IoU:")
        
        for class_idx, iou in iou_per_class.items():
            class_name = class_names[class_idx] if class_names else f"Class {class_idx}"
            print(f"  {class_name:20s}: {iou:.4f} ({iou*100:.2f}%)")
        
        print("="*70)


def calculate_iou(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> float:
    """
    Quick IoU calculation for a single batch
    
    Args:
        pred: Predictions [B, C, H, W] or [B, H, W]
        target: Ground truth [B, H, W]
        num_classes: Number of classes
    
    Returns:
        Mean IoU
    """
    metrics = SegmentationMetrics(num_classes)
    metrics.update(pred, target)
    return metrics.get_miou()


def calculate_pixel_accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Quick pixel accuracy calculation
    
    Args:
        pred: Predictions [B, C, H, W] or [B, H, W]
        target: Ground truth [B, H, W]
    
    Returns:
        Pixel accuracy
    """
    if pred.dim() == 4:
        pred = torch.argmax(pred, dim=1)
    
    correct = (pred == target).sum().item()
    total = target.numel()
    
    return correct / total if total > 0 else 0.0


# ═══════════════════════════════════════════════════════════════════
# TEST CODE
# ═══════════════════════════════════════════════════════════════════

def test_metrics():
    """Test metrics calculation"""
    print("Testing Segmentation Metrics...")
    
    # Create dummy predictions and targets
    num_classes = 12
    batch_size = 2
    height, width = 360, 480
    
    # Random predictions and targets
    predictions = torch.randint(0, num_classes, (batch_size, height, width))
    targets = torch.randint(0, num_classes, (batch_size, height, width))
    
    # Initialize metrics
    metrics = SegmentationMetrics(num_classes)
    
    # Update metrics
    metrics.update(predictions, targets)
    
    # Get metrics
    print(f"\nTest Results:")
    print(f"  mIoU: {metrics.get_miou():.4f}")
    print(f"  Pixel Accuracy: {metrics.get_pixel_accuracy():.4f}")
    print(f"  Dice Score: {metrics.get_dice_score():.4f}")
    
    # Print per-class IoU
    iou_per_class = metrics.get_iou_per_class()
    print(f"\nPer-class IoU:")
    for class_idx, iou in iou_per_class.items():
        print(f"  Class {class_idx}: {iou:.4f}")
    
    print("\n✓ Metrics test passed!")


if __name__ == "__main__":
    test_metrics()
