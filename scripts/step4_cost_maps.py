"""
Step 4: Generate Cost Maps for All Images
Converts segmentation predictions to navigation cost maps

Usage:
    python scripts/step4_cost_maps.py
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))
from utils.cost_map import CostMapGenerator

# Paths
predictions_dir = Path("results/predictions")
output_dir = Path("results/step4_cost_maps")
output_dir.mkdir(parents=True, exist_ok=True)

# Find all prediction masks
prediction_files = sorted(list(predictions_dir.glob("*_mask.png")))

print(f"\n{'='*70}")
print("STEP 4: GENERATING COST MAPS")
print(f"{'='*70}\n")
print(f"Found {len(prediction_files)} prediction masks\n")

# Initialize cost map generator
generator = CostMapGenerator()

# Process each prediction
for pred_file in tqdm(prediction_files, desc="Generating cost maps"):
    # Load segmentation mask
    seg_mask = np.array(Image.open(pred_file))
    
    # Generate cost map
    cost_map = generator.generate(seg_mask)
    
    # Save visualization
    output_file = output_dir / f"{pred_file.stem.replace('_mask', '')}_costmap.png"
    
    plt.figure(figsize=(10, 6))
    plt.imshow(cost_map, cmap='RdYlGn_r', vmin=0, vmax=1)
    plt.colorbar(label='Navigation Cost')
    plt.title('Cost Map (Green=Navigable, Red=Obstacles)')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

print(f"\n✓ Cost maps saved to: {output_dir}")
print(f"{'='*70}\n")
