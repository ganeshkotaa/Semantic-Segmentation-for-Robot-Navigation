"""
Step 6: Final Visualization - Path on Original Images
Overlays planned paths on original images

Usage:
    python scripts/step6_final_overlay.py
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2

sys.path.append(str(Path(__file__).parent.parent))
from utils.cost_map import CostMapGenerator
from utils.path_planning import AStarPlanner

# Paths
test_images_dir = Path("data/raw/camvid/test")
predictions_dir = Path("results/predictions")
output_dir = Path("results/step6_final_overlay")
output_dir.mkdir(parents=True, exist_ok=True)

# Find all images
prediction_files = sorted(list(predictions_dir.glob("*_mask.png")))

print(f"\n{'='*70}")
print("STEP 6: FINAL VISUALIZATION - PATH ON ORIGINAL IMAGES")
print(f"{'='*70}\n")
print(f"Found {len(prediction_files)} images\n")

# Initialize
generator = CostMapGenerator()

# Process each image
for pred_file in tqdm(prediction_files, desc="Creating final overlays"):
    # Find original image
    image_name = pred_file.stem.replace('_mask', '') + '.png'
    original_image_path = test_images_dir / image_name
    
    if not original_image_path.exists():
        continue
    
    # Load original image
    original_image = np.array(Image.open(original_image_path).convert('RGB'))
    
    # Load segmentation and generate cost map
    seg_mask = np.array(Image.open(pred_file))
    cost_map = generator.generate(seg_mask)
    
    # Initialize planner with this cost map
    planner = AStarPlanner(cost_map)
    
    # Define start and goal
    start = (20, 20)
    goal = (cost_map.shape[0] - 20, cost_map.shape[1] - 20)
    
    # Plan path
    path = planner.plan(start, goal)
    
    # Resize original image to match cost map if needed
    if original_image.shape[:2] != cost_map.shape:
        original_image = cv2.resize(original_image, 
                                    (cost_map.shape[1], cost_map.shape[0]))
    
    # Save visualization
    output_file = output_dir / f"{image_name.replace('.png', '')}_final.png"
    
    plt.figure(figsize=(12, 8))
    plt.imshow(original_image)
    
    if path and len(path) > 0:
        path_array = np.array(path)
        # Draw path with thick blue line
        plt.plot(path_array[:, 1], path_array[:, 0], 'b-', linewidth=4, 
                label='Navigation Path', alpha=0.8)
        # Draw start and goal markers
        plt.scatter([start[1]], [start[0]], c='lime', s=300, marker='o', 
                   edgecolors='black', linewidths=3, label='Start', zorder=5)
        plt.scatter([goal[1]], [goal[0]], c='red', s=300, marker='X', 
                   edgecolors='black', linewidths=3, label='Goal', zorder=5)
    
    plt.title('Final Navigation Path Overlay', fontsize=14, fontweight='bold')
    plt.legend(fontsize=12)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

print(f"\n✓ Final overlays saved to: {output_dir}")
print(f"{'='*70}\n")
