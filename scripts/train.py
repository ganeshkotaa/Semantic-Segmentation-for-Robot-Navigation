"""
Training script for DeepLabV3+ semantic segmentation
Complete training pipeline with validation, checkpointing, and logging

Usage:
    python scripts/train.py
"""

import os
import sys
from pathlib import Path
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import config
from models.deeplabv3plus import create_model
from utils.dataset import create_dataloaders
from utils.metrics import SegmentationMetrics
from utils.visualization import visualize_batch, plot_training_curves


class Trainer:
    """
    Trainer class for semantic segmentation
    
    Handles:
    - Training loop
    - Validation loop
    - Checkpointing
    - TensorBoard logging
    - Learning rate scheduling
    - Mixed precision training
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler._LRScheduler,
        criterion: nn.Module,
        device: str,
        num_classes: int,
        checkpoint_dir: Path,
        tensorboard_dir: Path,
        use_amp: bool = True
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.num_classes = num_classes
        self.checkpoint_dir = checkpoint_dir
        self.tensorboard_dir = tensorboard_dir
        self.use_amp = use_amp
        
        # Mixed precision scaler
        self.scaler = GradScaler() if use_amp else None
        
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=str(tensorboard_dir))
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_ious = []
        self.val_ious = []
        self.best_val_iou = 0.0
        self.current_epoch = 0
        
        # Early stopping
        self.patience = config.PATIENCE
        self.patience_counter = 0
        
        print(f"\n{'='*70}")
        print("TRAINER INITIALIZED")
        print(f"{'='*70}")
        print(f"Device: {device}")
        print(f"Mixed Precision: {use_amp}")
        print(f"Checkpoint Dir: {checkpoint_dir}")
        print(f"TensorBoard Dir: {tensorboard_dir}")
        print(f"{'='*70}\n")
    
    def train_epoch(self) -> tuple:
        """
        Train for one epoch
        
        Returns:
            (avg_loss, avg_iou)
        """
        self.model.train()
        
        epoch_loss = 0.0
        metrics = SegmentationMetrics(self.num_classes)
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1} [Train]")
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs['out'], labels)
                    
                    # Add auxiliary loss if available
                    if 'aux' in outputs:
                        aux_loss = self.criterion(outputs['aux'], labels)
                        loss = loss + 0.4 * aux_loss
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if config.CLIP_GRAD_NORM > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.CLIP_GRAD_NORM)
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs['out'], labels)
                
                if 'aux' in outputs:
                    aux_loss = self.criterion(outputs['aux'], labels)
                    loss = loss + 0.4 * aux_loss
                
                loss.backward()
                
                if config.CLIP_GRAD_NORM > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.CLIP_GRAD_NORM)
                
                self.optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            metrics.update(outputs['out'], labels)
            
            # Update progress bar
            current_iou = metrics.get_miou()
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'mIoU': f'{current_iou:.4f}'
            })
            
            # Log to TensorBoard
            if batch_idx % config.LOG_INTERVAL == 0:
                global_step = self.current_epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/BatchLoss', loss.item(), global_step)
        
        # Calculate epoch metrics
        avg_loss = epoch_loss / len(self.train_loader)
        avg_iou = metrics.get_miou()
        
        return avg_loss, avg_iou
    
    @torch.no_grad()
    def validate(self) -> tuple:
        """
        Validate model
        
        Returns:
            (avg_loss, avg_iou, metrics_dict)
        """
        self.model.eval()
        
        epoch_loss = 0.0
        metrics = SegmentationMetrics(self.num_classes)
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch + 1} [Val]")
        
        for batch in pbar:
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs['out'], labels)
            
            # Update metrics
            epoch_loss += loss.item()
            metrics.update(outputs['out'], labels)
            
            # Update progress bar
            current_iou = metrics.get_miou()
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'mIoU': f'{current_iou:.4f}'
            })
        
        # Calculate epoch metrics
        avg_loss = epoch_loss / len(self.val_loader)
        all_metrics = metrics.get_all_metrics()
        
        return avg_loss, all_metrics['miou'], all_metrics
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_iou': self.best_val_iou,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_ious': self.train_ious,
            'val_ious': self.val_ious
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        save_path = self.checkpoint_dir / filename
        torch.save(checkpoint, save_path)
        print(f"✓ Saved checkpoint: {filename}")
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint"""
        load_path = self.checkpoint_dir / filename
        
        if not load_path.exists():
            print(f"✗ Checkpoint not found: {filename}")
            return False
        
        checkpoint = torch.load(load_path)
        
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_iou = checkpoint['best_val_iou']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.train_ious = checkpoint['train_ious']
        self.val_ious = checkpoint['val_ious']
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"✓ Loaded checkpoint: {filename}")
        print(f"  Resuming from epoch {self.current_epoch + 1}")
        print(f"  Best val IoU: {self.best_val_iou:.4f}")
        
        return True
    
    def train(self, num_epochs: int):
        """
        Main training loop
        
        Args:
            num_epochs: Number of epochs to train
        """
        print(f"\n{'='*70}")
        print(f"STARTING TRAINING")
        print(f"{'='*70}")
        print(f"Total epochs: {num_epochs}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Train
            train_loss, train_iou = self.train_epoch()
            
            # Validate
            val_loss, val_iou, val_metrics = self.validate()
            
            # Update scheduler
            self.scheduler.step()
            
            # Record history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_ious.append(train_iou)
            self.val_ious.append(val_iou)
            
            # Log to TensorBoard
            self.writer.add_scalar('Epoch/TrainLoss', train_loss, epoch)
            self.writer.add_scalar('Epoch/ValLoss', val_loss, epoch)
            self.writer.add_scalar('Epoch/TrainIoU', train_iou, epoch)
            self.writer.add_scalar('Epoch/ValIoU', val_iou, epoch)
            self.writer.add_scalar('Epoch/LearningRate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start_time
            print(f"\n{'='*70}")
            print(f"EPOCH {epoch + 1}/{num_epochs} SUMMARY")
            print(f"{'='*70}")
            print(f"Time: {epoch_time:.2f}s")
            print(f"Train Loss: {train_loss:.4f} | Train mIoU: {train_iou:.4f}")
            print(f"Val Loss:   {val_loss:.4f} | Val mIoU:   {val_iou:.4f}")
            print(f"Val Pixel Acc: {val_metrics['pixel_accuracy']:.4f}")
            print(f"Val Dice: {val_metrics['dice_score']:.4f}")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save checkpoint
            if config.SAVE_CHECKPOINTS:
                # Save latest
                self.save_checkpoint('latest.pth')
                
                # Save best
                if val_iou > self.best_val_iou:
                    self.best_val_iou = val_iou
                    self.save_checkpoint('best.pth')
                    print(f"🎉 New best model! Val mIoU: {val_iou:.4f}")
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                
                # Save periodic checkpoints
                if (epoch + 1) % config.CHECKPOINT_FREQUENCY == 0:
                    self.save_checkpoint(f'epoch_{epoch+1}.pth')
            
            # Early stopping
            if config.EARLY_STOPPING and self.patience_counter >= self.patience:
                print(f"\n⚠ Early stopping triggered after {epoch + 1} epochs")
                print(f"No improvement for {self.patience} epochs")
                break
            
            print(f"{'='*70}\n")
        
        # Training finished
        total_time = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"TRAINING COMPLETED")
        print(f"{'='*70}")
        print(f"Total time: {total_time/60:.2f} minutes")
        print(f"Best val mIoU: {self.best_val_iou:.4f}")
        print(f"{'='*70}\n")
        
        # Close TensorBoard writer
        self.writer.close()
        
        # Plot training curves
        plot_training_curves(
            self.train_losses,
            self.val_losses,
            self.train_ious,
            self.val_ious,
            save_path=str(config.RESULTS_DIR / 'training_curves.png')
        )


def main():
    """Main training function"""
    print(f"\n{'='*70}")
    print("SEMANTIC SEGMENTATION FOR ROBOT NAVIGATION")
    print("Training Pipeline")
    print(f"{'='*70}\n")
    
    # Set random seeds for reproducibility
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.RANDOM_SEED)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print()
    
    # Create dataloaders
    print("Loading dataset...")
    data_dir = config.RAW_DATA_DIR / "camvid"
    
    dataloaders = create_dataloaders(
        data_dir=str(data_dir),
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        img_size=config.INPUT_SIZE
    )
    
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    
    print(f"✓ Dataset loaded")
    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Val samples: {len(val_loader.dataset)}")
    print()
    
    # Create model
    print("Creating model...")
    model = create_model(
        num_classes=config.NUM_CLASSES,
        pretrained=config.PRETRAINED_BACKBONE,
        device=device
    )
    print()
    
    # Create optimizer
    print("Setting up optimizer...")
    optimizer = optim.Adam(
        model.get_params(lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    print(f"✓ Optimizer: Adam")
    print(f"  Learning rate: {config.LEARNING_RATE}")
    print(f"  Weight decay: {config.WEIGHT_DECAY}")
    print()
    
    # Create scheduler
    print("Setting up learning rate scheduler...")
    if config.LR_SCHEDULER == 'poly':
        # Polynomial decay
        lambda_poly = lambda epoch: (1 - epoch / config.NUM_EPOCHS) ** config.LR_POWER
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_poly)
    elif config.LR_SCHEDULER == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.LR_STEP_SIZE,
            gamma=config.LR_GAMMA
        )
    else:  # cosine
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.NUM_EPOCHS
        )
    print(f"✓ Scheduler: {config.LR_SCHEDULER}")
    print()
    
    # Create loss function
    print("Setting up loss function...")
    criterion = nn.CrossEntropyLoss()
    print(f"✓ Loss function: CrossEntropyLoss")
    print()
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        num_classes=config.NUM_CLASSES,
        checkpoint_dir=config.CHECKPOINTS_DIR,
        tensorboard_dir=config.TENSORBOARD_DIR,
        use_amp=config.USE_AMP
    )
    
    # Start training
    trainer.train(num_epochs=config.NUM_EPOCHS)
    
    print("\n✓ Training complete!")
    print(f"\nNext steps:")
    print(f"1. View training curves: tensorboard --logdir={config.TENSORBOARD_DIR}")
    print(f"2. Evaluate model: python scripts/evaluate.py")
    print(f"3. Run inference: See notebooks/03_inference.ipynb")


if __name__ == "__main__":
    main()
