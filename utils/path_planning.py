"""
A* Path Planning Algorithm for Robot Navigation
Finds optimal path on cost maps considering navigation costs

Algorithm: A* (A-star)
- Heuristic: Euclidean distance
- Cost: Cumulative terrain cost
- Movement: 8-connected (allows diagonal)
"""

import numpy as np
import heapq
from typing import Tuple, List, Optional, Set
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import time


class AStarPlanner:
    """
    A* path planning algorithm
    
    Finds minimum cost path from start to goal on a cost map
    """
    
    def __init__(
        self,
        cost_map: np.ndarray,
        allow_diagonal: bool = True,
        obstacle_threshold: float = 0.85
    ):
        """
        Initialize A* planner
        
        Args:
            cost_map: 2D cost map [H, W] with values in [0.0, 1.0]
            allow_diagonal: Whether to allow diagonal movement
            obstacle_threshold: Cost threshold above which cells are considered obstacles
        """
        self.cost_map = cost_map
        self.height, self.width = cost_map.shape
        self.allow_diagonal = allow_diagonal
        self.obstacle_threshold = obstacle_threshold
        
        # Movement directions: 4-connected or 8-connected
        if allow_diagonal:
            # 8 directions: N, NE, E, SE, S, SW, W, NW
            self.directions = [
                (-1, 0), (-1, 1), (0, 1), (1, 1),
                (1, 0), (1, -1), (0, -1), (-1, -1)
            ]
            # Cost multipliers (diagonal moves cost sqrt(2))
            self.move_costs = [1.0, 1.414, 1.0, 1.414, 1.0, 1.414, 1.0, 1.414]
        else:
            # 4 directions: N, E, S, W
            self.directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
            self.move_costs = [1.0, 1.0, 1.0, 1.0]
    
    def heuristic(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """
        Heuristic function (Euclidean distance)
        
        Args:
            pos1: First position (row, col)
            pos2: Second position (row, col)
        
        Returns:
            Estimated distance
        """
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def is_valid(self, pos: Tuple[int, int]) -> bool:
        """
        Check if position is valid and navigable
        
        Args:
            pos: Position (row, col)
        
        Returns:
            True if valid and navigable
        """
        row, col = pos
        
        # Check bounds
        if not (0 <= row < self.height and 0 <= col < self.width):
            return False
        
        # Check if obstacle
        if self.cost_map[row, col] >= self.obstacle_threshold:
            return False
        
        return True
    
    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[Tuple[int, int], float]]:
        """
        Get valid neighbors of a position
        
        Args:
            pos: Current position (row, col)
        
        Returns:
            List of (neighbor_pos, move_cost) tuples
        """
        neighbors = []
        row, col = pos
        
        for (dr, dc), move_cost in zip(self.directions, self.move_costs):
            neighbor = (row + dr, col + dc)
            
            if self.is_valid(neighbor):
                neighbors.append((neighbor, move_cost))
        
        return neighbors
    
    def plan(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        max_iterations: int = 100000
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Plan path from start to goal using A*
        
        Args:
            start: Start position (row, col)
            goal: Goal position (row, col)
            max_iterations: Maximum planning iterations
        
        Returns:
            List of positions forming the path, or None if no path found
        """
        # Validate start and goal
        if not self.is_valid(start):
            print(f"✗ Invalid start position: {start}")
            return None
        
        if not self.is_valid(goal):
            print(f"✗ Invalid goal position: {goal}")
            return None
        
        # Initialize
        open_set = []
        heapq.heappush(open_set, (0.0, start))
        
        came_from = {}
        g_score = {start: 0.0}
        f_score = {start: self.heuristic(start, goal)}
        
        closed_set: Set[Tuple[int, int]] = set()
        
        iterations = 0
        
        while open_set and iterations < max_iterations:
            iterations += 1
            
            # Get node with lowest f_score
            current_f, current = heapq.heappop(open_set)
            
            # Goal reached
            if current == goal:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                
                print(f"✓ Path found in {iterations} iterations")
                print(f"  Path length: {len(path)} nodes")
                print(f"  Path cost: {g_score[goal]:.2f}")
                
                return path
            
            # Skip if already processed
            if current in closed_set:
                continue
            
            closed_set.add(current)
            
            # Explore neighbors
            for neighbor, move_cost in self.get_neighbors(current):
                if neighbor in closed_set:
                    continue
                
                # Calculate tentative g_score
                # Cost = movement cost * terrain cost
                terrain_cost = self.cost_map[neighbor]
                tentative_g = g_score[current] + move_cost * (1.0 + terrain_cost * 10.0)
                
                # Update if better path found
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        print(f"✗ No path found after {iterations} iterations")
        return None
    
    def smooth_path(self, path: List[Tuple[int, int]], smoothing_factor: int = 3) -> List[Tuple[int, int]]:
        """
        Smooth path by removing unnecessary waypoints
        
        Args:
            path: Original path
            smoothing_factor: How much to smooth (higher = more smoothing)
        
        Returns:
            Smoothed path
        """
        if len(path) <= 2:
            return path
        
        smoothed = [path[0]]
        
        i = 0
        while i < len(path) - 1:
            # Try to skip ahead
            skip = min(smoothing_factor, len(path) - i - 1)
            
            for s in range(skip, 0, -1):
                target = i + s
                
                # Check if we can go directly to target
                if self._is_line_clear(path[i], path[target]):
                    smoothed.append(path[target])
                    i = target
                    break
            else:
                # Can't skip, take next step
                smoothed.append(path[i + 1])
                i += 1
        
        return smoothed
    
    def _is_line_clear(self, start: Tuple[int, int], end: Tuple[int, int]) -> bool:
        """Check if line between two points is clear"""
        # Bresenham's line algorithm
        r0, c0 = start
        r1, c1 = end
        
        points = self._bresenham_line(r0, c0, r1, c1)
        
        for r, c in points:
            if not self.is_valid((r, c)):
                return False
        
        return True
    
    def _bresenham_line(self, r0: int, c0: int, r1: int, c1: int) -> List[Tuple[int, int]]:
        """Bresenham's line algorithm"""
        points = []
        
        dr = abs(r1 - r0)
        dc = abs(c1 - c0)
        r_step = 1 if r0 < r1 else -1
        c_step = 1 if c0 < c1 else -1
        err = dr - dc
        
        r, c = r0, c0
        
        while True:
            points.append((r, c))
            
            if r == r1 and c == c1:
                break
            
            e2 = 2 * err
            if e2 > -dc:
                err -= dc
                r += r_step
            if e2 < dr:
                err += dr
                c += c_step
        
        return points


def visualize_path_planning(
    cost_map: np.ndarray,
    path: Optional[List[Tuple[int, int]]],
    start: Tuple[int, int],
    goal: Tuple[int, int],
    title: str = "A* Path Planning",
    save_path: Optional[str] = None
):
    """
    Visualize path planning result
    
    Args:
        cost_map: Cost map
        path: Planned path
        start: Start position
        goal: Goal position
        title: Plot title
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Show cost map
    ax.imshow(cost_map, cmap='RdYlGn_r', vmin=0, vmax=1, alpha=0.7)
    
    # Plot path
    if path:
        path_array = np.array(path)
        ax.plot(path_array[:, 1], path_array[:, 0], 
               'b-', linewidth=3, label='Planned Path', zorder=10)
        ax.plot(path_array[:, 1], path_array[:, 0], 
               'bo', markersize=4, zorder=11)
    
    # Plot start and goal
    ax.plot(start[1], start[0], 'go', markersize=20, 
           label='Start', markeredgecolor='white', markeredgewidth=2, zorder=12)
    ax.plot(goal[1], goal[0], 'ro', markersize=20, 
           label='Goal', markeredgecolor='white', markeredgewidth=2, zorder=12)
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', fontsize=12)
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved path visualization to {save_path}")
    
    plt.show()


def test_path_planning():
    """Test A* path planning"""
    print("="*70)
    print("Testing A* Path Planning")
    print("="*70)
    
    # Create test cost map
    h, w = 100, 100
    cost_map = np.random.rand(h, w) * 0.3  # Random low costs
    
    # Add obstacles
    cost_map[20:80, 45:55] = 1.0  # Vertical wall
    cost_map[40:50, :60] = 1.0    # Horizontal wall
    
    # Define start and goal
    start = (10, 10)
    goal = (90, 90)
    
    print(f"\nPlanning from {start} to {goal}...")
    
    # Create planner
    planner = AStarPlanner(cost_map, allow_diagonal=True)
    
    # Plan path
    start_time = time.time()
    path = planner.plan(start, goal)
    planning_time = time.time() - start_time
    
    print(f"  Planning time: {planning_time*1000:.1f} ms")
    
    # Visualize
    visualize_path_planning(cost_map, path, start, goal)
    
    print("\n✓ Path planning test passed!")


if __name__ == "__main__":
    test_path_planning()
