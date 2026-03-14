"""
Complete Navigation Demo
Demonstrates the full pipeline: Segmentation → Cost Map → Path Planning

This script showcases the complete robot navigation system:
1. Load trained segmentation model
2. Segment input image
3. Generate navigation cost map
4. Plan optimal path using A*
5. Create professional visualizations

Usage:
    python scripts/demo_navigation.py --checkpoint models/checkpoints/best.pth --image path/to/image.jpg
"""

import os
import sys
from pathlib import Path
import argparse
from typing import Tuple, List
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
import torch

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import config
from scripts.inference import SegmentationInference
from utils.cost_map import CostMapGenerator, create_cost_map_from_prediction
from utils.path_planning import AStarPlanner, visualize_path_planning
from utils.visualization import denormalize_image, create_color_map, label_to_color


class NavigationDemo:
    """
    Complete navigation demonstration system
    
    Integrates segmentation, cost mapping, and path planning
    """
    
    def __init__(self, checkpoint_path: str, device: str = 'cuda'):
        """
        Initialize navigation demo
        
        Args:
            checkpoint_path: Path to trained model
            device: Device to run on
        """
        print(f"\n{'='*70}")
        print("INITIALIZING NAVIGATION SYSTEM")
        print(f"{'='*70}\n")
        
        # Initialize inference engine
        self.inference = SegmentationInference(
            checkpoint_path=checkpoint_path,
            device=device
        )
        
        # Initialize cost map generator
        self.cost_generator = CostMapGenerator()
        
        print("✓ Navigation system ready\n")
    
    def process_image(
        self,
        image_path: str,
        start_pos: Tuple[int, int],
        goal_pos: Tuple[int, int],
        downsample_size: Tuple[int, int] = (100, 100),
        use_uncertainty: bool = True
    ) -> dict:
        """
        Process single image through full navigation pipeline
        
        Args:
            image_path: Path to input image
            start_pos: Start position (row, col) in downsampled coordinates
            goal_pos: Goal position (row, col) in downsampled coordinates
            downsample_size: Size for path planning grid
            use_uncertainty: Whether to use prediction confidence in cost map
        
        Returns:
            Dictionary with all results
        """
        print(f"\n{'='*70}")
        print(f"PROCESSING: {Path(image_path).name}")
        print(f"{'='*70}\n")
        
        # Step 1: Semantic Segmentation
        print("[1/4] Running semantic segmentation...")
        result = self.inference.predict(image_path)
        prediction = result['prediction']
        confidence = result['confidence']
        
        print(f"✓ Segmentation complete")
        print(f"  Unique classes: {np.unique(prediction).tolist()}")
        print(f"  Mean confidence: {confidence.mean():.3f}")
        
        # Step 2: Generate Cost Map
        print(f"\n[2/4] Generating navigation cost map...")
        
        if use_uncertainty:
            cost_map_full = self.cost_generator.generate_with_uncertainty(
                prediction, confidence
            )
        else:
            cost_map_full = self.cost_generator.generate(prediction)
        
        # Downsample for path planning
        cost_map = self.cost_generator.downsample(cost_map_full, downsample_size)
        
        stats = self.cost_generator.get_statistics(cost_map)
        print(f"✓ Cost map generated")
        print(f"  Size: {cost_map.shape}")
        print(f"  Free area: {stats['free_area_pct']:.1f}%")
        print(f"  Obstacle area: {stats['obstacle_area_pct']:.1f}%")
        
        # Step 3: Path Planning
        print(f"\n[3/4] Planning path from {start_pos} to {goal_pos}...")
        
        planner = AStarPlanner(
            cost_map=cost_map,
            allow_diagonal=True,
            obstacle_threshold=0.85
        )
        
        path = planner.plan(start_pos, goal_pos)
        
        if path:
            # Smooth path
            smoothed_path = planner.smooth_path(path, smoothing_factor=3)
            print(f"✓ Path planning successful")
            print(f"  Original path: {len(path)} nodes")
            print(f"  Smoothed path: {len(smoothed_path)} nodes")
        else:
            print(f"✗ Path planning failed - no valid path found")
            smoothed_path = None
        
        # Step 4: Load original image
        print(f"\n[4/4] Preparing visualizations...")
        original_image = np.array(Image.open(image_path).convert('RGB'))
        
        # Resize prediction to match original if needed
        if original_image.shape[:2] != prediction.shape:
            from scipy.ndimage import zoom
            scale_h = original_image.shape[0] / prediction.shape[0]
            scale_w = original_image.shape[1] / prediction.shape[1]
            prediction_resized = zoom(prediction, (scale_h, scale_w), order=0)
        else:
            prediction_resized = prediction
        
        print(f"✓ All processing complete\n")
        
        return {
            'original_image': original_image,
            'prediction': prediction,
            'prediction_resized': prediction_resized,
            'confidence': confidence,
            'cost_map_full': cost_map_full,
            'cost_map': cost_map,
            'path': path,
            'smoothed_path': smoothed_path,
            'start': start_pos,
            'goal': goal_pos,
            'statistics': stats
        }
    
    def create_comprehensive_visualization(
        self,
        results: dict,
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """
        Create comprehensive visualization showing all steps
        
        Args:
            results: Results dictionary from process_image
            save_path: Optional path to save figure
            show: Whether to display the figure
        """
        # Create figure with custom layout
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Get data
        original = results['original_image']
        prediction = results['prediction_resized']
        cost_map = results['cost_map']
        path = results['smoothed_path'] if results['smoothed_path'] else results['path']
        start = results['start']
        goal = results['goal']
        
        # Color map for segmentation
        color_map = create_color_map()
        pred_colored = label_to_color(prediction, color_map)
        
        # Row 1: Original, Segmentation, Overlay
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(original)
        ax1.set_title('(a) Input Image', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(pred_colored)
        ax2.set_title('(b) Semantic Segmentation', fontsize=14, fontweight='bold')
        ax2.axis('off')
        
        ax3 = fig.add_subplot(gs[0, 2])
        import cv2
        overlay = cv2.addWeighted(original, 0.6, pred_colored, 0.4, 0)
        ax3.imshow(overlay)
        ax3.set_title('(c) Segmentation Overlay', fontsize=14, fontweight='bold')
        ax3.axis('off')
        
        # Row 2: Cost Map, Path on Cost Map, Path on Image
        ax4 = fig.add_subplot(gs[1, 0])
        im1 = ax4.imshow(cost_map, cmap='RdYlGn_r', vmin=0, vmax=1)
        ax4.set_title('(d) Navigation Cost Map', fontsize=14, fontweight='bold')
        ax4.axis('off')
        plt.colorbar(im1, ax=ax4, fraction=0.046, label='Cost')
        
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.imshow(cost_map, cmap='RdYlGn_r', vmin=0, vmax=1, alpha=0.7)
        if path:
            path_array = np.array(path)
            ax5.plot(path_array[:, 1], path_array[:, 0], 'b-', linewidth=3, label='Path')
            ax5.plot(start[1], start[0], 'go', markersize=15, label='Start',
                    markeredgecolor='white', markeredgewidth=2)
            ax5.plot(goal[1], goal[0], 'ro', markersize=15, label='Goal',
                    markeredgecolor='white', markeredgewidth=2)
            ax5.legend(loc='upper right')
        ax5.set_title('(e) A* Path Planning', fontsize=14, fontweight='bold')
        ax5.axis('off')
        
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.imshow(original)
        if path:
            # Scale path to original image size
            scale_h = original.shape[0] / cost_map.shape[0]
            scale_w = original.shape[1] / cost_map.shape[1]
            path_scaled = [(int(r * scale_h), int(c * scale_w)) for r, c in path]
            path_array = np.array(path_scaled)
            ax6.plot(path_array[:, 1], path_array[:, 0], 'y-', linewidth=4, 
                    label='Planned Path', alpha=0.8)
            ax6.plot(path_array[:, 1], path_array[:, 0], 'b-', linewidth=2)
            
            start_scaled = (int(start[0] * scale_h), int(start[1] * scale_w))
            goal_scaled = (int(goal[0] * scale_h), int(goal[1] * scale_w))
            ax6.plot(start_scaled[1], start_scaled[0], 'go', markersize=20,
                    markeredgecolor='white', markeredgewidth=3)
            ax6.plot(goal_scaled[1], goal_scaled[0], 'ro', markersize=20,
                    markeredgecolor='white', markeredgewidth=3)
        ax6.set_title('(f) Path on Original Image', fontsize=14, fontweight='bold')
        ax6.axis('off')
        
        # Row 3: Statistics and legend
        ax7 = fig.add_subplot(gs[2, :])
        ax7.axis('off')
        
        # Add statistics text
        stats = results['statistics']
        stats_text = f"""
        Navigation System Results:
        
        Segmentation: {len(np.unique(results['prediction']))} classes detected
        Confidence: {results['confidence'].mean():.1%} average confidence
        
        Cost Map:
        • Free area (navigable): {stats['free_area_pct']:.1f}%
        • Caution area: {stats['caution_area_pct']:.1f}%
        • Avoid area: {stats['avoid_area_pct']:.1f}%
        • Obstacle area: {stats['obstacle_area_pct']:.1f}%
        
        Path Planning:
        • Algorithm: A* with Euclidean heuristic
        • Path length: {len(path) if path else 0} nodes
        • Status: {'✓ Success' if path else '✗ No path found'}
        """
        
        ax7.text(0.05, 0.5, stats_text, transform=ax7.transAxes,
                fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                family='monospace')
        
        # Add class legend
        from matplotlib.patches import Patch
        legend_elements = []
        class_names = list(config.CAMVID_CLASSES.values())
        unique_classes = np.unique(results['prediction'])
        
        for class_idx in unique_classes[:6]:  # Show first 6 classes
            if class_idx < len(class_names):
                color = color_map[class_idx] / 255.0
                legend_elements.append(
                    Patch(facecolor=color, label=class_names[class_idx])
                )
        
        ax7.legend(handles=legend_elements, loc='center right',
                  title='Detected Classes', frameon=True, fontsize=10)
        
        # Main title
        fig.suptitle('Robot Navigation System: Segmentation + Path Planning',
                    fontsize=18, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"✓ Saved comprehensive visualization to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()


def run_multiple_demos(
    inference_engine: NavigationDemo,
    test_scenarios: List[dict],
    output_dir: Path
):
    """
    Run multiple navigation demonstrations
    
    Args:
        inference_engine: NavigationDemo instance
        test_scenarios: List of scenario dictionaries
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"RUNNING {len(test_scenarios)} TEST SCENARIOS")
    print(f"{'='*70}\n")
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n--- Scenario {i}/{len(test_scenarios)} ---")
        
        results = inference_engine.process_image(
            image_path=scenario['image'],
            start_pos=scenario['start'],
            goal_pos=scenario['goal'],
            downsample_size=scenario.get('grid_size', (100, 100))
        )
        
        # Save visualization
        save_path = output_dir / f"scenario_{i}_{Path(scenario['image']).stem}.png"
        inference_engine.create_comprehensive_visualization(
            results=results,
            save_path=str(save_path),
            show=False
        )


def main():
    """Main demo function"""
    parser = argparse.ArgumentParser(description='Complete navigation system demo')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to input image')
    parser.add_argument('--output-dir', type=str, default='results/navigation_demos',
                        help='Output directory')
    parser.add_argument('--start', type=int, nargs=2, default=[10, 10],
                        help='Start position (row col)')
    parser.add_argument('--goal', type=int, nargs=2, default=[90, 90],
                        help='Goal position (row col)')
    parser.add_argument('--grid-size', type=int, nargs=2, default=[100, 100],
                        help='Planning grid size (height width)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu)')
    parser.add_argument('--run-multiple', action='store_true',
                        help='Run multiple test scenarios')
    
    args = parser.parse_args()
    
    # Set device
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # Create navigation demo
    demo = NavigationDemo(
        checkpoint_path=args.checkpoint,
        device=device
    )
    
    output_dir = Path(args.output_dir)
    
    if args.run_multiple:
        # Define test scenarios
        # Note: Update these with actual test image paths after training
        test_scenarios = [
            {
                'image': 'data/raw/camvid/test/0001TP_006690.png',
                'start': (90, 10),
                'goal': (10, 90),
                'grid_size': (100, 100)
            },
            {
                'image': 'data/raw/camvid/test/0001TP_007500.png',
                'start': (80, 20),
                'goal': (20, 80),
                'grid_size': (100, 100)
            },
            {
                'image': 'data/raw/camvid/test/0001TP_008850.png',
                'start': (70, 30),
                'goal': (30, 70),
                'grid_size': (100, 100)
            }
        ]
        
        run_multiple_demos(demo, test_scenarios, output_dir)
        
    elif args.image:
        # Single image demo
        start = tuple(args.start)
        goal = tuple(args.goal)
        grid_size = tuple(args.grid_size)
        
        results = demo.process_image(
            image_path=args.image,
            start_pos=start,
            goal_pos=goal,
            downsample_size=grid_size
        )
        
        # Create visualization
        save_path = output_dir / f"navigation_{Path(args.image).stem}.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        demo.create_comprehensive_visualization(
            results=results,
            save_path=str(save_path),
            show=True
        )
        
        print(f"\n{'='*70}")
        print("DEMO COMPLETE")
        print(f"{'='*70}")
        print(f"\n✓ Results saved to {save_path}")
    
    else:
        print("\n✗ Please provide --image or use --run-multiple")


if __name__ == "__main__":
    main()
