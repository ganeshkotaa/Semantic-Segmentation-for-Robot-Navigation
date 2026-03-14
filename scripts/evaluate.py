"""
Evaluation script for trained segmentation model
Tests on validation/test set and generates predictions

Usage:
    python scripts/evaluate.py --checkpoint models/checkpoints/best.pth --split test
"""

import os
import sys
from pathlib import Path
import argparse

import torch
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import config
from models.deeplabv3plus import create_model
from utils.dataset import create_dataloaders
from utils.metrics import SegmentationMetrics
from utils.visualization import visualize_batch, visualize_prediction


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    num_classes: int,
    save_predictions: bool = True,
    max_visualizations: int = 10
):
    """
    Evaluate model on dataset
    
    Args:
        model: Trained model
        dataloader: DataLoader for evaluation
        device: Device to run on
        num_classes: Number of classes
        save_predictions: Whether to save prediction visualizations
        max_visualizations: Maximum number of predictions to visualize
    
    Returns:
        Dictionary with metrics
    """
    model.eval()
    
    metrics = SegmentationMetrics(num_classes)
    
    # Create predictions directory
    if save_predictions:
        pred_dir = config.PREDICTIONS_DIR
        pred_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"\n{'='*70}")
    print("EVALUATING MODEL")
    print(f"{'='*70}")
    
    pbar = tqdm(dataloader, desc="Evaluating")
    
    viz_count = 0
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        # Forward pass
        outputs = model(images)
        predictions = outputs['out']
        
        # Update metrics
        metrics.update(predictions, labels)
        
        # Save visualizations
        if save_predictions and viz_count < max_visualizations:
            batch_size = min(images.shape[0], max_visualizations - viz_count)
            
            for i in range(batch_size):
                save_path = pred_dir / f"prediction_{batch_idx}_{i}.png"
                
                visualize_prediction(
                    image=images[i],
                    prediction=predictions[i],
                    target=labels[i],
                    class_names=list(config.CAMVID_CLASSES.values()),
                    save_path=str(save_path),
                    show=False
                )
                
                viz_count += 1
        
        # Update progress bar
        current_metrics = metrics.get_all_metrics()
        pbar.set_postfix({
            'mIoU': f'{current_metrics["miou"]:.4f}',
            'PixAcc': f'{current_metrics["pixel_accuracy"]:.4f}'
        })
    
    # Get final metrics
    final_metrics = metrics.get_all_metrics()
    iou_per_class = metrics.get_iou_per_class()
    
    # Print results
    print(f"\n{'='*70}")
    print("EVALUATION RESULTS")
    print(f"{'='*70}")
    
    print(f"\nOverall Metrics:")
    print(f"  Mean IoU:        {final_metrics['miou']:.4f} ({final_metrics['miou']*100:.2f}%)")
    print(f"  Pixel Accuracy:  {final_metrics['pixel_accuracy']:.4f} ({final_metrics['pixel_accuracy']*100:.2f}%)")
    print(f"  Dice Score:      {final_metrics['dice_score']:.4f} ({final_metrics['dice_score']*100:.2f}%)")
    
    print(f"\nPer-Class IoU:")
    class_names = list(config.CAMVID_CLASSES.values())
    for class_idx, iou in iou_per_class.items():
        class_name = class_names[class_idx] if class_idx < len(class_names) else f"Class {class_idx}"
        print(f"  {class_name:20s}: {iou:.4f} ({iou*100:.2f}%)")
    
    print(f"\n{'='*70}")
    
    if save_predictions:
        print(f"\n✓ Saved {viz_count} prediction visualizations to {pred_dir}")
    
    return final_metrics


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Evaluate semantic segmentation model')
    parser.add_argument('--checkpoint', type=str, default='models/checkpoints/best.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'],
                        help='Dataset split to evaluate on')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size (default: from config)')
    parser.add_argument('--save-predictions', action='store_true', default=True,
                        help='Save prediction visualizations')
    parser.add_argument('--max-visualizations', type=int, default=20,
                        help='Maximum number of predictions to visualize')
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print("SEMANTIC SEGMENTATION MODEL EVALUATION")
    print(f"{'='*70}\n")
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # Load model
    print("Loading model...")
    model = create_model(
        num_classes=config.NUM_CLASSES,
        pretrained=False,  # Don't need pretrained since we're loading checkpoint
        device=device
    )
    
    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        # Try relative to project root
        checkpoint_path = Path(__file__).parent.parent / args.checkpoint
    
    if not checkpoint_path.exists():
        print(f"✗ Checkpoint not found: {args.checkpoint}")
        print(f"Please train a model first: python scripts/train.py")
        return
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    if 'epoch' in checkpoint:
        print(f"  Epoch: {checkpoint['epoch'] + 1}")
    if 'best_val_iou' in checkpoint:
        print(f"  Best val IoU: {checkpoint['best_val_iou']:.4f}")
    print()
    
    # Load dataset
    print(f"Loading {args.split} dataset...")
    data_dir = config.RAW_DATA_DIR / "camvid"
    
    batch_size = args.batch_size if args.batch_size else config.BATCH_SIZE
    
    dataloaders = create_dataloaders(
        data_dir=str(data_dir),
        batch_size=batch_size,
        num_workers=config.NUM_WORKERS,
        img_size=config.INPUT_SIZE
    )
    
    dataloader = dataloaders[args.split]
    print(f"✓ Loaded {len(dataloader.dataset)} samples")
    print()
    
    # Evaluate
    metrics = evaluate_model(
        model=model,
        dataloader=dataloader,
        device=device,
        num_classes=config.NUM_CLASSES,
        save_predictions=args.save_predictions,
        max_visualizations=args.max_visualizations
    )
    
    print("\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()
