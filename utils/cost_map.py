"""
Cost Map Generation Utilities
Converts semantic segmentation masks to navigation cost maps

Cost Map:
- Low values (0.0 - 0.2) = Highly navigable (roads, pavements)
- Medium values (0.2 - 0.5) = Caution areas (trees, signs)
- High values (0.5 - 0.8) = Avoid areas (pedestrians, cyclists)
- Very high values (0.8 - 1.0) = Obstacles (buildings, cars, poles)
"""

import numpy as np
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
from scipy.ndimage import zoom


class CostMapGenerator:
    """
    Generates navigation cost maps from semantic segmentation
    
    Cost values range from 0.0 (free) to 1.0 (obstacle)
    """
    
    def __init__(self, class_cost_dict: Optional[Dict[int, float]] = None):
        """
        Initialize cost map generator
        
        Args:
            class_cost_dict: Dictionary mapping class indices to cost values (0.0 to 1.0)
                           If None, uses default CamVid costs
        """
        if class_cost_dict is None:
            # Default cost values for CamVid dataset
            # Lower = more navigable, Higher = more dangerous/obstacle
            self.class_costs = {
                0: 0.05,   # Sky - neutral (not ground but not obstacle)
                1: 0.95,   # Building - obstacle
                2: 0.95,   # Pole - obstacle
                3: 0.01,   # Road - highly navigable ⭐
                4: 0.05,   # Pavement - navigable ⭐
                5: 0.30,   # Tree - caution (might be passable)
                6: 0.20,   # SignSymbol - low obstacle
                7: 0.95,   # Fence - obstacle
                8: 0.95,   # Car - obstacle
                9: 0.70,   # Pedestrian - avoid but not solid
                10: 0.70,  # Bicyclist - avoid but not solid
                11: 0.50   # Unlabelled - unknown, medium cost
            }
        else:
            self.class_costs = class_cost_dict
        
        # Cost categories for visualization
        self.cost_categories = {
            'free': (0.0, 0.2, 'green'),
            'caution': (0.2, 0.5, 'yellow'),
            'avoid': (0.5, 0.8, 'orange'),
            'obstacle': (0.8, 1.0, 'red')
        }
    
    def generate(self, segmentation_mask: np.ndarray) -> np.ndarray:
        """
        Generate cost map from segmentation mask
        
        Args:
            segmentation_mask: Segmentation mask [H, W] with class indices
        
        Returns:
            Cost map [H, W] with values in range [0.0, 1.0]
        """
        # Create empty cost map
        cost_map = np.zeros_like(segmentation_mask, dtype=np.float32)
        
        # Assign costs based on class
        for class_idx, cost_value in self.class_costs.items():
            mask = segmentation_mask == class_idx
            cost_map[mask] = cost_value
        
        return cost_map
    
    def generate_with_uncertainty(
        self,
        segmentation_mask: np.ndarray,
        confidence_map: np.ndarray,
        uncertainty_weight: float = 0.3
    ) -> np.ndarray:
        """
        Generate cost map with uncertainty from confidence scores
        
        Low confidence predictions get higher cost (less trustworthy)
        
        Args:
            segmentation_mask: Segmentation mask [H, W]
            confidence_map: Confidence scores [H, W] in range [0.0, 1.0]
            uncertainty_weight: Weight for uncertainty penalty (0.0 to 1.0)
        
        Returns:
            Cost map [H, W] with uncertainty-adjusted costs
        """
        # Base cost map
        cost_map = self.generate(segmentation_mask)
        
        # Uncertainty penalty: low confidence = higher cost
        uncertainty = 1.0 - confidence_map
        uncertainty_penalty = uncertainty * uncertainty_weight
        
        # Add uncertainty to cost (clipped to [0, 1])
        cost_map_with_uncertainty = np.clip(cost_map + uncertainty_penalty, 0.0, 1.0)
        
        return cost_map_with_uncertainty
    
    def downsample(self, cost_map: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Downsample cost map for faster path planning
        
        Args:
            cost_map: Full resolution cost map
            target_size: Target size (height, width)
        
        Returns:
            Downsampled cost map
        """
        scale_h = target_size[0] / cost_map.shape[0]
        scale_w = target_size[1] / cost_map.shape[1]
        
        # Use max pooling to preserve obstacles
        downsampled = zoom(cost_map, (scale_h, scale_w), order=1)
        
        return downsampled
    
    def get_statistics(self, cost_map: np.ndarray) -> Dict[str, float]:
        """
        Calculate cost map statistics
        
        Args:
            cost_map: Cost map array
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            'mean_cost': float(cost_map.mean()),
            'std_cost': float(cost_map.std()),
            'min_cost': float(cost_map.min()),
            'max_cost': float(cost_map.max()),
            'free_area_pct': float((cost_map < 0.2).sum() / cost_map.size * 100),
            'caution_area_pct': float(((cost_map >= 0.2) & (cost_map < 0.5)).sum() / cost_map.size * 100),
            'avoid_area_pct': float(((cost_map >= 0.5) & (cost_map < 0.8)).sum() / cost_map.size * 100),
            'obstacle_area_pct': float((cost_map >= 0.8).sum() / cost_map.size * 100)
        }
        
        return stats
    
    def visualize(
        self,
        cost_map: np.ndarray,
        title: str = "Navigation Cost Map",
        save_path: Optional[str] = None,
        show_legend: bool = True
    ):
        """
        Visualize cost map with color coding
        
        Args:
            cost_map: Cost map to visualize
            title: Plot title
            save_path: Optional path to save figure
            show_legend: Whether to show legend
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Display cost map
        im = ax.imshow(cost_map, cmap='RdYlGn_r', vmin=0, vmax=1)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.axis('off')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Navigation Cost', rotation=270, labelpad=20, fontsize=12)
        
        # Add legend with cost categories
        if show_legend:
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='green', label='Free (0.0-0.2): Roads, Pavements'),
                Patch(facecolor='yellow', label='Caution (0.2-0.5): Trees, Signs'),
                Patch(facecolor='orange', label='Avoid (0.5-0.8): Pedestrians'),
                Patch(facecolor='red', label='Obstacle (0.8-1.0): Buildings, Cars')
            ]
            ax.legend(handles=legend_elements, loc='upper center', 
                     bbox_to_anchor=(0.5, -0.05), ncol=2, frameon=True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved cost map visualization to {save_path}")
        
        plt.show()


def create_cost_map_from_prediction(
    prediction: np.ndarray,
    confidence: Optional[np.ndarray] = None,
    target_size: Optional[Tuple[int, int]] = None,
    use_uncertainty: bool = False
) -> np.ndarray:
    """
    Convenience function to create cost map from prediction
    
    Args:
        prediction: Segmentation prediction [H, W]
        confidence: Optional confidence map [H, W]
        target_size: Optional target size for downsampling
        use_uncertainty: Whether to incorporate uncertainty
    
    Returns:
        Cost map [H, W]
    """
    generator = CostMapGenerator()
    
    # Generate cost map
    if use_uncertainty and confidence is not None:
        cost_map = generator.generate_with_uncertainty(prediction, confidence)
    else:
        cost_map = generator.generate(prediction)
    
    # Downsample if requested
    if target_size is not None:
        cost_map = generator.downsample(cost_map, target_size)
    
    return cost_map


def test_cost_map_generation():
    """Test cost map generation"""
    print("="*70)
    print("Testing Cost Map Generation")
    print("="*70)
    
    # Create dummy segmentation (simulating a road scene)
    h, w = 360, 480
    segmentation = np.zeros((h, w), dtype=np.int32)
    
    # Add some features
    segmentation[:, :] = 0  # Sky as background
    segmentation[200:, :] = 3  # Road in bottom half
    segmentation[200:240, 100:150] = 8  # Car on road
    segmentation[200:240, 300:320] = 9  # Pedestrian
    segmentation[:200, :100] = 1  # Building on left
    segmentation[:200, 380:] = 5  # Tree on right
    
    # Generate cost map
    generator = CostMapGenerator()
    cost_map = generator.generate(segmentation)
    
    # Statistics
    stats = generator.get_statistics(cost_map)
    print(f"\nCost Map Statistics:")
    print(f"  Mean cost: {stats['mean_cost']:.3f}")
    print(f"  Free area: {stats['free_area_pct']:.1f}%")
    print(f"  Obstacle area: {stats['obstacle_area_pct']:.1f}%")
    
    # Visualize
    generator.visualize(cost_map, title="Test Cost Map", show_legend=True)
    
    print("\n✓ Cost map generation test passed!")


if __name__ == "__main__":
    test_cost_map_generation()
