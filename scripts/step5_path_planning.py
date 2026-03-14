"""
Step 5: A* Path Planning for All Images
Generates navigation paths on cost maps

Usage:
    python scripts/step5_path_planning.py
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))
from utils.cost_map import CostMapGenerator
from utils.path_planning import AStarPlanner

# Paths
predictions_dir = Path("results/predictions")
output_dir = Path("results/step5_path_planning")
output_dir.mkdir(parents=True, exist_ok=True)

# Find all prediction masks
prediction_files = sorted(list(predictions_dir.glob("*_mask.png")))

print(f"\n{'='*70}")
print("STEP 5: A* PATH PLANNING")
print(f"{'='*70}\n")
print(f"Found {len(prediction_files)} masks\n")

# Initialize
generator = CostMapGenerator()

# Process each prediction
for pred_file in tqdm(prediction_files, desc="Planning paths"):
    # Load segmentation and generate cost map
    seg_mask = np.array(Image.open(pred_file))
    cost_map = generator.generate(seg_mask)
    
    # Initialize planner with this cost map
    planner = AStarPlanner(cost_map, obstacle_threshold=0.95)  # Higher threshold
    
    # Define start and goal (top-left to bottom-right)
    start = (20, 20)
    goal = (cost_map.shape[0] - 20, cost_map.shape[1] - 20)
    
    # Make sure start/goal are navigable
    if cost_map[start[0], start[1]] > 0.5:
        # Find nearby navigable spot
        for r in range(max(0, start[0]-10), min(cost_map.shape[0], start[0]+10)):
            for c in range(max(0, start[1]-10), min(cost_map.shape[1], start[1]+10)):
                if cost_map[r, c] < 0.5:
                    start = (r, c)
                    break
    
    if cost_map[goal[0], goal[1]] > 0.5:
        # Find nearby navigable spot
        for r in range(max(0, goal[0]-10), min(cost_map.shape[0], goal[0]+10)):
            for c in range(max(0, goal[1]-10), min(cost_map.shape[1], goal[1]+10)):
                if cost_map[r, c] < 0.5:
                    goal = (r, c)
                    break
    
    # Plan path
    path = planner.plan(start, goal)
    
    # Save visualization
    output_file = output_dir / f"{pred_file.stem.replace('_mask', '')}_path.png"
    
    plt.figure(figsize=(10, 6))
    plt.imshow(cost_map, cmap='RdYlGn_r', vmin=0, vmax=1)
    
    if path and len(path) > 1:  # Need at least 2 points for a path
        path_array = np.array(path)
        plt.plot(path_array[:, 1], path_array[:, 0], 'b-', linewidth=3, label='Planned Path')
        plt.scatter([start[1], goal[1]], [start[0], goal[0]], 
                   c='yellow', s=200, marker='*', edgecolors='black', linewidths=2,
                   label='Start/Goal', zorder=5)
        plt.legend()
    else:
        # No path found, just show cost map
        plt.text(cost_map.shape[1]//2, 20, 'No path found', 
                ha='center', fontsize=12, color='red', weight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.title('A* Path Planning')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

print(f"\n✓ Paths saved to: {output_dir}")
print(f"{'='*70}\n")
