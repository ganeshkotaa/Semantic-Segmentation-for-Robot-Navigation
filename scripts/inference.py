"""
Inference Script for Semantic Segmentation
Loads trained model and performs inference on single images or batches

Usage:
    # Single image
    python scripts/inference.py --image path/to/image.jpg --checkpoint models/checkpoints/best.pth
    
    # Batch inference
    python scripts/inference.py --input-dir path/to/images/ --checkpoint models/checkpoints/best.pth
"""

import os
import sys
from pathlib import Path
import argparse
from typing import Union, List

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import config
from models.deeplabv3plus import create_model
from utils.visualization import visualize_prediction, denormalize_image, create_color_map, label_to_color


class SegmentationInference:
    """
    Inference class for semantic segmentation
    
    Handles model loading, preprocessing, and prediction
    """
    
    def __init__(self, checkpoint_path: str, device: str = 'cuda', num_classes: int = 12):
        """
        Initialize inference engine
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            device: Device to run inference on ('cuda' or 'cpu')
            num_classes: Number of segmentation classes
        """
        self.device = device
        self.num_classes = num_classes
        
        # Load model
        print(f"Loading model from {checkpoint_path}...")
        self.model = create_model(
            num_classes=num_classes,
            pretrained=False,
            device=device
        )
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.model.eval()
        
        # Get training info
        self.epoch = checkpoint.get('epoch', -1)
        self.best_iou = checkpoint.get('best_val_iou', -1)
        
        print(f"✓ Model loaded successfully")
        if self.epoch >= 0:
            print(f"  Trained for {self.epoch + 1} epochs")
        if self.best_iou >= 0:
            print(f"  Best validation IoU: {self.best_iou:.4f}")
        
        # Preprocessing transform
        self.transform = A.Compose([
            A.Resize(height=config.IMAGE_HEIGHT, width=config.IMAGE_WIDTH),
            A.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
            ToTensorV2()
        ])
    
    def preprocess_image(self, image: Union[str, np.ndarray, Image.Image]) -> torch.Tensor:
        """
        Preprocess image for inference
        
        Args:
            image: Input image (path, numpy array, or PIL Image)
        
        Returns:
            Preprocessed image tensor [1, 3, H, W]
        """
        # Load image if path
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        
        # Convert PIL to numpy
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Apply preprocessing
        transformed = self.transform(image=image)
        image_tensor = transformed['image']
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor
    
    @torch.no_grad()
    def predict(self, image: Union[str, np.ndarray, Image.Image]) -> dict:
        """
        Run inference on single image
        
        Args:
            image: Input image
        
        Returns:
            Dictionary with:
                - 'logits': Raw model output [1, C, H, W]
                - 'prediction': Class predictions [H, W]
                - 'probabilities': Class probabilities [C, H, W]
                - 'confidence': Confidence map [H, W]
        """
        # Preprocess
        image_tensor = self.preprocess_image(image).to(self.device)
        
        # Inference
        outputs = self.model(image_tensor)
        logits = outputs['out']  # [1, C, H, W]
        
        # Get predictions
        probabilities = F.softmax(logits, dim=1)[0]  # [C, H, W]
        prediction = torch.argmax(probabilities, dim=0)  # [H, W]
        confidence = torch.max(probabilities, dim=0)[0]  # [H, W]
        
        return {
            'logits': logits,
            'prediction': prediction.cpu().numpy(),
            'probabilities': probabilities.cpu().numpy(),
            'confidence': confidence.cpu().numpy()
        }
    
    def predict_batch(self, images: List[Union[str, np.ndarray, Image.Image]]) -> List[dict]:
        """
        Run inference on batch of images
        
        Args:
            images: List of input images
        
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for image in tqdm(images, desc="Running inference"):
            result = self.predict(image)
            results.append(result)
        
        return results


def save_predictions(
    image_path: str,
    prediction: np.ndarray,
    output_dir: Path,
    save_mask: bool = True,
    save_overlay: bool = True,
    save_colored: bool = True
):
    """
    Save prediction results in various formats
    
    Args:
        image_path: Path to original image
        prediction: Prediction mask [H, W]
        output_dir: Output directory
        save_mask: Save raw prediction mask
        save_overlay: Save overlay on original image
        save_colored: Save color-coded segmentation
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get base filename
    base_name = Path(image_path).stem
    
    # Load original image
    original_image = np.array(Image.open(image_path).convert('RGB'))
    
    # Resize prediction to match original if needed
    if original_image.shape[:2] != prediction.shape:
        from scipy.ndimage import zoom
        scale_h = original_image.shape[0] / prediction.shape[0]
        scale_w = original_image.shape[1] / prediction.shape[1]
        prediction = zoom(prediction, (scale_h, scale_w), order=0)
    
    # Save raw mask
    if save_mask:
        mask_path = output_dir / f"{base_name}_mask.png"
        Image.fromarray(prediction.astype(np.uint8)).save(mask_path)
    
    # Save colored segmentation
    if save_colored:
        color_map = create_color_map()
        colored = label_to_color(prediction, color_map)
        colored_path = output_dir / f"{base_name}_segmentation.png"
        Image.fromarray(colored).save(colored_path)
    
    # Save overlay
    if save_overlay:
        import cv2
        color_map = create_color_map()
        colored = label_to_color(prediction, color_map)
        overlay = cv2.addWeighted(original_image, 0.6, colored, 0.4, 0)
        overlay_path = output_dir / f"{base_name}_overlay.png"
        Image.fromarray(overlay).save(overlay_path)


def main():
    """Main inference function"""
    parser = argparse.ArgumentParser(description='Run semantic segmentation inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to single input image')
    parser.add_argument('--input-dir', type=str, default=None,
                        help='Directory with input images')
    parser.add_argument('--output-dir', type=str, default='results/inference',
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu)')
    parser.add_argument('--visualize', action='store_true',
                        help='Show visualizations')
    
    args = parser.parse_args()
    
    # Set device
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"\n{'='*70}")
    print("SEMANTIC SEGMENTATION INFERENCE")
    print(f"{'='*70}")
    print(f"Device: {device}")
    
    # Create inference engine
    inference = SegmentationInference(
        checkpoint_path=args.checkpoint,
        device=device,
        num_classes=config.NUM_CLASSES
    )
    
    # Prepare output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Single image inference
    if args.image:
        print(f"\nProcessing single image: {args.image}")
        
        result = inference.predict(args.image)
        prediction = result['prediction']
        confidence = result['confidence']
        
        print(f"✓ Inference complete")
        print(f"  Prediction shape: {prediction.shape}")
        print(f"  Unique classes: {np.unique(prediction).tolist()}")
        print(f"  Mean confidence: {confidence.mean():.3f}")
        
        # Save results
        save_predictions(
            image_path=args.image,
            prediction=prediction,
            output_dir=output_dir
        )
        
        print(f"\n✓ Results saved to {output_dir}")
    
    # Batch inference
    elif args.input_dir:
        input_dir = Path(args.input_dir)
        
        # Find all images
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(input_dir.glob(f"*{ext}")))
            image_files.extend(list(input_dir.glob(f"*{ext.upper()}")))
        
        print(f"\nFound {len(image_files)} images in {input_dir}")
        
        if len(image_files) == 0:
            print("No images found!")
            return
        
        # Process all images
        for img_path in tqdm(image_files, desc="Processing images"):
            result = inference.predict(str(img_path))
            prediction = result['prediction']
            
            save_predictions(
                image_path=str(img_path),
                prediction=prediction,
                output_dir=output_dir
            )
        
        print(f"\n✓ Processed {len(image_files)} images")
        print(f"✓ Results saved to {output_dir}")
    
    else:
        print("\n✗ Please provide either --image or --input-dir")
        return
    
    print(f"\n{'='*70}")
    print("INFERENCE COMPLETE")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
