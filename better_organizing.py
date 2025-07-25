import numpy as np
from scipy.optimize import linear_sum_assignment
import math
from rtree import index
from collections import defaultdict
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import itertools
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
import time

class ParallelMotionOptimizer:
    def __init__(self, grid_size, min_separation, movement_speed=1.0):
        self.grid_size = grid_size
        self.min_separation = min_separation
        self.speed = movement_speed
        self.spots = set()
        self.time_steps = None
        self.n_cores = mp.cpu_count()

    @staticmethod
    def check_pair_collision(args):
        """Static method for parallel collision checking"""
        path1, path2, min_separation, time_steps, speed = args
        start1, end1 = path1
        start2, end2 = path2
        
        for t in range(time_steps + 1):
            t_normalized = t / time_steps
            # Interpolate positions
            pos1 = (
                start1[0] + (end1[0] - start1[0]) * t_normalized,
                start1[1] + (end1[1] - start1[1]) * t_normalized
            )
            pos2 = (
                start2[0] + (end2[0] - start2[0]) * t_normalized,
                start2[1] + (end2[1] - start2[1]) * t_normalized
            )
            
            # Check distance
            dist = math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
            if dist < min_separation:
                return True
        return False

    def euclidean_distance(self, p1, p2):
        return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

    def find_movement_groups_parallel(self, assignments):
        """Parallel version of movement group finding"""
        n_spots = len(assignments)
        
        # Generate all pairs of paths to check
        pairs = list(itertools.combinations(range(n_spots), 2))
        
        # Prepare arguments for parallel processing
        check_args = [
            (
                (assignments[i][0], assignments[i][1]),
                (assignments[j][0], assignments[j][1]),
                self.min_separation,
                20,  # time_steps
                self.speed
            )
            for i, j in pairs
        ]
        
        # Parallel collision checking
        movement_graph = defaultdict(set)
        with ProcessPoolExecutor(max_workers=self.n_cores) as executor:
            collision_results = list(executor.map(
                self.check_pair_collision, 
                check_args, 
                chunksize=max(1, len(check_args) // (self.n_cores * 4))
            ))
        
        # Build movement graph from results
        for (i, j), collides in zip(pairs, collision_results):
            if collides:
                movement_graph[i].add(j)
                movement_graph[j].add(i)

        # Parallel graph coloring
        return self.parallel_graph_coloring(movement_graph, n_spots)

    def parallel_graph_coloring(self, movement_graph, n_spots):
        """Parallel graph coloring algorithm"""
        def color_subset(node_subset):
            local_colors = {}
            for node in node_subset:
                used_colors = {
                    local_colors.get(neighbor) 
                    for neighbor in movement_graph[node] 
                    if neighbor in local_colors
                }
                for color in range(n_spots):
                    if color not in used_colors:
                        local_colors[node] = color
                        break
            return local_colors

        # Split nodes into subsets for parallel processing
        nodes = list(range(n_spots))
        subset_size = max(1, n_spots // self.n_cores)
        node_subsets = [
            nodes[i:i + subset_size] 
            for i in range(0, n_spots, subset_size)
        ]

        # Color subsets in parallel
        with ProcessPoolExecutor(max_workers=self.n_cores) as executor:
            partial_colorings = list(executor.map(color_subset, node_subsets))

        # Merge colorings
        final_colors = {}
        for partial in partial_colorings:
            final_colors.update(partial)

        # Resolve conflicts between subsets
        with ProcessPoolExecutor(max_workers=self.n_cores) as executor:
            conflict_resolutions = list(executor.map(
                lambda n: self.resolve_conflicts(n, final_colors, movement_graph),
                range(n_spots)
            ))

        final_colors.update(dict(enumerate(conflict_resolutions)))

        # Group spots by color
        movement_groups = defaultdict(list)
        assignments = list(self.spots)  # Convert to list for indexing
        for spot_idx, color in final_colors.items():
            movement_groups[color].append(assignments[spot_idx])

        return movement_groups
    
    @staticmethod
    def resolve_conflicts(node, colors, graph):
        """Resolve coloring conflicts for a single node"""
        used_colors = {colors[neighbor] for neighbor in graph[node]}
        current_color = colors.get(node)
        
        if current_color is None or current_color in used_colors:
            for color in itertools.count():
                if color not in used_colors:
                    return color
        return current_color

    def optimize_parallel_movement(self, target_pattern):
        """Main optimization function with parallel processing"""
        if len(target_pattern) != len(self.spots):
            raise ValueError(f"Target pattern has {len(target_pattern)} points but there are {len(self.spots)} spots")
        
        pattern = self.scale_pattern(target_pattern)
        assignments = self.assign_spots_to_targets(pattern)
        
        # Parallel movement group finding
        movement_groups = self.find_movement_groups_parallel(assignments)
        
        # Calculate execution times (can be parallelized for large groups)
        if len(movement_groups) > 1000:  # Only parallelize for large problems
            with ProcessPoolExecutor(max_workers=self.n_cores) as executor:
                group_times = dict(executor.map(
                    lambda g: (g[0], self.calculate_group_time(g[1])),
                    movement_groups.items()
                ))
        else:
            group_times = {
                group_id: self.calculate_group_time(paths)
                for group_id, paths in movement_groups.items()
            }
        
        return movement_groups, group_times

    def calculate_group_time(self, paths):
        """Calculate execution time for a group of paths"""
        max_distance = max(
            math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            for start, end in paths
        )
        return max_distance / self.speed

    def simulate_movement_parallel(self, movement_groups, time_steps=50):
        """Parallel version of movement simulation"""
        max_time = max(
            math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2) / self.speed
            for group in movement_groups.values()
            for start, end in group
        )
        
        def calculate_frame(t):
            t_normalized = t / (time_steps - 1)
            positions = []
            
            for group in movement_groups.values():
                for start, end in group:
                    distance = math.sqrt(
                        (end[0] - start[0])**2 + (end[1] - start[1])**2
                    )
                    time_needed = distance / self.speed
                    progress = min(t_normalized * max_time / time_needed, 1.0)
                    
                    pos = (
                        start[0] + (end[0] - start[0]) * progress,
                        start[1] + (end[1] - start[1]) * progress
                    )
                    positions.append(pos)
            
            return positions

        # Parallel frame calculation
        with ProcessPoolExecutor(max_workers=self.n_cores) as executor:
            all_positions = list(executor.map(
                calculate_frame, 
                range(time_steps),
                chunksize=max(1, time_steps // (self.n_cores * 4))
            ))
        
        return all_positions

    def scale_pattern(self, pattern):
        """Scale and center the pattern to fit the grid"""
        pattern = np.array(pattern)
        
        min_x, min_y = pattern.min(axis=0)
        max_x, max_y = pattern.max(axis=0)
        
        current_width = max_x - min_x
        current_height = max_y - min_y
        scale = min(
            (self.grid_size * 0.8) / current_width,
            (self.grid_size * 0.8) / current_height
        )
        
        pattern = pattern * scale
        pattern_center = pattern.mean(axis=0)
        grid_center = np.array([self.grid_size/2, self.grid_size/2])
        pattern = pattern - pattern_center + grid_center
        
        return [tuple(p) for p in pattern]

    def assign_spots_to_targets(self, target_positions):
        """Assign spots to target positions using Hungarian algorithm"""
        cost_matrix = np.zeros((len(self.spots), len(target_positions)))
        spots_list = list(self.spots)
        
        for i, spot in enumerate(spots_list):
            for j, target in enumerate(target_positions):
                cost_matrix[i, j] = self.euclidean_distance(spot, target)
        
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        return [(spots_list[i], target_positions[j]) for i, j in zip(row_ind, col_ind)]
    
    def visualize_movement(grid_size, all_positions, min_separation):
        """Visualize the movement of spots"""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        def init():
            ax.clear()
            ax.set_xlim(-1, grid_size + 1)
            ax.set_ylim(-1, grid_size + 1)
            return []
        
        def animate(frame):
            ax.clear()
            ax.set_xlim(-1, grid_size + 1)
            ax.set_ylim(-1, grid_size + 1)
            
            # Draw spots
            positions = all_positions[frame]
            x = [p[0] for p in positions]
            y = [p[1] for p in positions]
            
            # Draw spots and their separation circles
            for px, py in zip(x, y):
                ax.add_artist(plt.Circle((px, py), min_separation/2, 
                            fill=False, linestyle='--', alpha=0.3))
                ax.plot(px, py, 'bo')
                
            return []
    
        anim = FuncAnimation(fig, animate, init_func=init, 
                        frames=len(all_positions), 
                        interval=50, blit=True)
        return anim

    def save_trajectories_to_csv(all_positions, decimal_places=3, filename="spot_trajectories.csv"):
        """Save spot trajectories to CSV file with rounded values"""
        # Initialize lists for each spot's coordinates
        n_spots = len(all_positions[0])
        n_timesteps = len(all_positions)
        
        # Create column names
        columns = []
        for spot_idx in range(n_spots):
            columns.extend([f'spot_{spot_idx}_x', f'spot_{spot_idx}_y'])
        
        # Create data dictionary
        data = {col: [] for col in columns}
        
        # Fill data with rounded values
        for timestep in range(n_timesteps):
            for spot_idx in range(n_spots):
                x, y = all_positions[timestep][spot_idx]
                data[f'spot_{spot_idx}_x'].append(round(x, decimal_places))
                data[f'spot_{spot_idx}_y'].append(round(y, decimal_places))
        
        # Create and save DataFrame
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"Trajectories saved to {filename}")
    
    def create_rectangular_grid_pattern(n_spots, grid_size, spot_spacing, initial_positions=None, use_com=True):
        """
        Create a rectangular grid pattern for the target positions.
        
        Parameters:
        n_spots (int): Number of spots to arrange
        grid_size (float): Size of the overall grid
        spot_spacing (float): Minimum space between spots in the grid
        initial_positions (list): List of (x,y) tuples of initial spot positions
        use_com (bool): If True, center pattern on center of mass of initial positions
        
        Returns:
        list of tuples: Target positions arranged in a rectangular grid
        """
        # Calculate the number of rows and columns needed
        n_cols = int(math.sqrt(n_spots))
        n_rows = math.ceil(n_spots / n_cols)
        
        # Calculate the total width and height of the grid
        total_width = (n_cols - 1) * spot_spacing
        total_height = (n_rows - 1) * spot_spacing
        
        if use_com and initial_positions:
            # Calculate center of mass of initial positions
            com_x = sum(x for x, _ in initial_positions) / len(initial_positions)
            com_y = sum(y for _, y in initial_positions) / len(initial_positions)
            
            # Ensure the pattern stays within grid boundaries
            start_x = max(spot_spacing, min(grid_size - total_width - spot_spacing,
                        com_x - total_width/2))
            start_y = max(spot_spacing, min(grid_size - total_height - spot_spacing,
                        com_y - total_height/2))
        else:
            # Center the pattern in the grid
            start_x = (grid_size - total_width) / 2
            start_y = (grid_size - total_height) / 2
        
        # Generate the positions
        target_pattern = []
        spot_count = 0
        
        for row in range(n_rows):
            for col in range(n_cols):
                if spot_count < n_spots:
                    x = start_x + col * spot_spacing
                    y = start_y + row * spot_spacing
                    target_pattern.append((x, y))
                    spot_count += 1
        
        return target_pattern

    def calculate_path_metrics(initial_positions, target_positions, movement_groups):
        """
        Calculate metrics for the paths.
        
        Returns:
        dict: Contains various path metrics
        """
        metrics = {
            'total_distance': 0,
            'average_distance': 0,
            'max_distance': 0,
            'min_distance': float('inf'),
            'path_lengths': []
        }
        
        for i, (start, end) in enumerate(zip(initial_positions, target_positions)):
            distance = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            metrics['path_lengths'].append(distance)
            metrics['total_distance'] += distance
            metrics['max_distance'] = max(metrics['max_distance'], distance)
            metrics['min_distance'] = min(metrics['min_distance'], distance)
        
        metrics['average_distance'] = metrics['total_distance'] / len(initial_positions)
        metrics['std_deviation'] = np.std(metrics['path_lengths'])
        
        return metrics

    def main():
        # Setup parameters
        grid_size = 12
        min_separation = 1.0
        movement_speed = 2.0
        n_spots = 100
        
        # Create optimizer
        optimizer = ParallelMotionOptimizer(grid_size, min_separation, movement_speed)
        
        # Generate random initial spots biased towards one side
        while len(optimizer.spots) < n_spots:
            # Bias towards left side for demonstration
            x = np.random.uniform(0, grid_size/2)  # Only use left half of grid
            y = np.random.uniform(0, grid_size)
            new_spot = (x, y)
            if all(math.sqrt((x - ex)**2 + (y - ey)**2) >= min_separation 
                for ex, ey in optimizer.spots):
                optimizer.spots.add(new_spot)
        
        # Create two versions of the target pattern
        target_pattern_centered = create_rectangular_grid_pattern(
            n_spots=n_spots,
            grid_size=grid_size,
            spot_spacing=min_separation * 1.1,
            initial_positions=list(optimizer.spots),
            use_com=False
        )
        
        target_pattern_com = create_rectangular_grid_pattern(
            n_spots=n_spots,
            grid_size=grid_size,
            spot_spacing=min_separation * 1.1,
            initial_positions=list(optimizer.spots),
            use_com=True
        )
        
        # Compare optimization times and movement groups for both approaches
        print("Testing center-fixed pattern:")
        start_time = time.time()
        groups_centered, times_centered = optimizer.optimize_parallel_movement(target_pattern_centered)
        center_time = time.time() - start_time
        
        print(f"Center-fixed pattern completed in {center_time:.2f} seconds")
        print(f"Number of movement groups: {len(groups_centered)}")
        
        # Reset optimizer spots
        original_spots = optimizer.spots.copy()
        optimizer.spots = original_spots
        
        print("\nTesting center-of-mass pattern:")
        start_time = time.time()
        groups_com, times_com = optimizer.optimize_parallel_movement(target_pattern_com)
        com_time = time.time() - start_time
        
        print(f"Center-of-mass pattern completed in {com_time:.2f} seconds")
        print(f"Number of movement groups: {len(groups_com)}")
        
        return {
            'center_fixed': {
                'time': center_time,
                'groups': len(groups_centered),
                'pattern': target_pattern_centered
            },
            'center_of_mass': {
                'time': com_time,
                'groups': len(groups_com),
                'pattern': target_pattern_com
            }
        }

    if __name__ == "__main__":
        results = main()