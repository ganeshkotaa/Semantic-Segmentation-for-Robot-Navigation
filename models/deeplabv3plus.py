"""
DeepLabV3+ Model Implementation for Semantic Segmentation
Uses pretrained ResNet-50 backbone from torchvision
Modified for custom number of classes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from typing import Dict


class DeepLabV3Plus(nn.Module):
    """
    DeepLabV3+ semantic segmentation model
    
    Architecture:
    - Backbone: ResNet-50 (pretrained on ImageNet)
    - ASPP (Atrous Spatial Pyramid Pooling) module
    - Decoder with low-level features
    - Final classification layer
    
    Args:
        num_classes: Number of segmentation classes
        pretrained: Whether to use pretrained backbone
        output_stride: Output stride (8 or 16)
    """
    
    def __init__(
        self,
        num_classes: int = 12,
        pretrained: bool = True,
        output_stride: int = 16
    ):
        super(DeepLabV3Plus, self).__init__()
        
        self.num_classes = num_classes
        
        # Load pretrained DeepLabV3 from torchvision
        if pretrained:
            print("Loading pretrained DeepLabV3-ResNet50...")
            weights = DeepLabV3_ResNet50_Weights.DEFAULT
            base_model = deeplabv3_resnet50(weights=weights)
        else:
            base_model = deeplabv3_resnet50(weights=None)
        
        # Extract backbone and classifier
        self.backbone = base_model.backbone
        
        # Modify classifier for our number of classes
        # Original classifier has 21 classes (COCO + background)
        # We replace the final conv layer
        in_channels = base_model.classifier[4].in_channels
        
        # Build new classifier head
        self.classifier = nn.Sequential(
            base_model.classifier[0],  # ASPP module
            base_model.classifier[1],  # Conv
            base_model.classifier[2],  # BatchNorm
            base_model.classifier[3],  # ReLU
            nn.Conv2d(in_channels, num_classes, kernel_size=1)  # Final conv
        )
        
        # Auxiliary classifier (helps training)
        if hasattr(base_model, 'aux_classifier') and base_model.aux_classifier is not None:
            aux_in_channels = base_model.aux_classifier[4].in_channels
            self.aux_classifier = nn.Sequential(
                base_model.aux_classifier[0],
                base_model.aux_classifier[1],
                base_model.aux_classifier[2],
                base_model.aux_classifier[3],
                nn.Conv2d(aux_in_channels, num_classes, kernel_size=1)
            )
        else:
            self.aux_classifier = None
        
        print(f"✓ DeepLabV3+ initialized with {num_classes} classes")
        print(f"  - Pretrained: {pretrained}")
        print(f"  - Backbone: ResNet-50")
        print(f"  - Output stride: {output_stride}")
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor [B, 3, H, W]
        
        Returns:
            Dictionary with 'out' (main output) and optionally 'aux' (auxiliary output)
        """
        input_shape = x.shape[-2:]
        
        # Extract features from backbone
        features = self.backbone(x)
        
        # Main output
        x_main = features['out']
        x_main = self.classifier(x_main)
        x_main = F.interpolate(x_main, size=input_shape, mode='bilinear', align_corners=False)
        
        result = {'out': x_main}
        
        # Auxiliary output (if available and training)
        if self.aux_classifier is not None and self.training:
            x_aux = features['aux']
            x_aux = self.aux_classifier(x_aux)
            x_aux = F.interpolate(x_aux, size=input_shape, mode='bilinear', align_corners=False)
            result['aux'] = x_aux
        
        return result
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict segmentation mask
        
        Args:
            x: Input tensor [B, 3, H, W]
        
        Returns:
            Segmentation mask [B, H, W] with class indices
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
            prediction = torch.argmax(output['out'], dim=1)
        return prediction
    
    def get_params(self, lr: float, weight_decay: float = 0.0):
        """
        Get parameters for optimizer with different learning rates
        
        Args:
            lr: Base learning rate
            weight_decay: Weight decay
        
        Returns:
            List of parameter groups
        """
        # Backbone parameters (lower learning rate)
        backbone_params = list(self.backbone.parameters())
        
        # Classifier parameters (higher learning rate)
        classifier_params = list(self.classifier.parameters())
        if self.aux_classifier is not None:
            classifier_params += list(self.aux_classifier.parameters())
        
        return [
            {'params': backbone_params, 'lr': lr * 0.1, 'weight_decay': weight_decay},
            {'params': classifier_params, 'lr': lr, 'weight_decay': weight_decay}
        ]


def create_model(num_classes: int = 12, pretrained: bool = True, device: str = 'cuda') -> DeepLabV3Plus:
    """
    Create DeepLabV3+ model
    
    Args:
        num_classes: Number of segmentation classes
        pretrained: Whether to use pretrained backbone
        device: Device to load model on
    
    Returns:
        DeepLabV3Plus model
    """
    model = DeepLabV3Plus(
        num_classes=num_classes,
        pretrained=pretrained
    )
    
    model = model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Information:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Device: {device}")
    
    return model


def test_model():
    """Test model creation and forward pass"""
    print("="*70)
    print("Testing DeepLabV3+ Model")
    print("="*70)
    
    # Create model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = create_model(num_classes=12, pretrained=True, device=device)
    
    # Test forward pass
    batch_size = 2
    height, width = 360, 480
    
    print(f"\nTesting forward pass...")
    print(f"  Input shape: [{batch_size}, 3, {height}, {width}]")
    
    # Create dummy input
    x = torch.randn(batch_size, 3, height, width).to(device)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(x)
    
    print(f"\n✓ Forward pass successful!")
    print(f"  Output shape: {output['out'].shape}")
    print(f"  Expected shape: [{batch_size}, 12, {height}, {width}]")
    
    # Test prediction
    prediction = model.predict(x)
    print(f"\n✓ Prediction successful!")
    print(f"  Prediction shape: {prediction.shape}")
    print(f"  Expected shape: [{batch_size}, {height}, {width}]")
    print(f"  Unique classes: {torch.unique(prediction).tolist()}")
    
    print("\n" + "="*70)
    print("✓ MODEL TEST PASSED")
    print("="*70)


if __name__ == "__main__":
    test_model()
