"""
Batch Navigation Demo
Run complete navigation pipeline on all test images

Usage:
    python scripts/batch_navigation_demo.py --checkpoint models/checkpoints/best.pth
"""

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from scripts.demo_navigation import NavigationDemo
import config


def main():
    parser = argparse.ArgumentParser(description='Batch Navigation Demo')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--input-dir', type=str, 
                        default='data/raw/camvid/test',
                        help='Directory with input images')
    parser.add_argument('--output-dir', type=str,
                        default='results/batch_navigation',
                        help='Output directory')
    parser.add_argument('--num-images', type=int, default=10,
                        help='Number of images to process (default: 10)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use (cpu or cuda)')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("BATCH NAVIGATION DEMO")
    print("="*70 + "\n")
    
    # Get input images
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"✗ Input directory not found: {input_dir}")
        return
    
    image_files = sorted(list(input_dir.glob("*.png")))[:args.num_images]
    
    if len(image_files) == 0:
        print(f"✗ No images found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} images")
    print(f"Processing first {min(args.num_images, len(image_files))} images\n")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize demo
    demo = NavigationDemo(
        checkpoint_path=args.checkpoint,
        device=args.device
    )
    
    # Process each image
    print("Processing images...\n")
    for i, image_path in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}] {image_path.name}")
        
        try:
            # Process image
            result = demo.process_image(str(image_path))
            
            # Create visualization
            output_path = output_dir / f"{image_path.stem}_navigation.png"
            demo.create_comprehensive_visualization(
                result,
                save_path=str(output_path)
            )
            
            print(f"  ✓ Saved to {output_path.name}\n")
            
        except Exception as e:
            print(f"  ✗ Error: {e}\n")
            continue
    
    print("="*70)
    print("✓ BATCH PROCESSING COMPLETE!")
    print(f"Output directory: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
