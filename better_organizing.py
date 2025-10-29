from math import ceil, sqrt
import numpy as np
from scipy.optimize import linear_sum_assignment
import csv
import random

# better_organizing.py
"""
BetterOrganizer

Purpose:
- Read 2D point coordinates (list or CSV).
- Create a rectangular grid centered on the mean position of the points.
- Assign points to grid nodes using the Hungarian algorithm (optimal assignment).
- Create straight-line paths from each point to its assigned grid node.
- Sample each path into increments of a given length (step_length).
- Simulate stepping along paths in synchronized time steps.
- Detect collisions (euclidean distance below a threshold) and attempt to resolve them
    by (a) reassigning with perturbed grid nodes, and (b) adding small start delays (stagger)
    if necessary.

Usage (example):
        org = BetterOrganizer(points_array, step_length=0.5, collision_distance=0.4)
        result = org.organize()
        # result contains:
        #  - "assignments": array of shape (N,2) target coordinates for each agent
        #  - "indices": array mapping agent -> target index in grid_nodes
        #  - "trajectories": array shape (N, T, 2): positions per global timestep (padded with final pos)
        #  - "delays": list of integer delay-steps applied per agent
        #  - "success": boolean whether a collision-free plan was found
"""





class BetterOrganizer:
        def __init__(
                self,
                points,
                grid_shape=None,
                spacing=None,
                step_length=0.5,
                collision_distance=0.5,
                max_attempts=50,
                max_delay_steps=8,
                perturb_scale=0.25,
                random_seed=None,
        ):
                """
                points: array-like shape (N,2) or list of (x,y)
                grid_shape: tuple (rows, cols) or None (auto near-square)
                spacing: distance between adjacent grid nodes; if None, computed from points extents
                step_length: sampling increment length along straight-line paths
                collision_distance: minimum allowed distance between any two agents at same timestep
                max_attempts: number of reassignment attempts (with perturbations) before giving up
                max_delay_steps: max number of discrete step delays to try per-agent to avoid collisions
                perturb_scale: fraction of spacing used to perturb grid nodes when reassigning
                random_seed: optional int for reproducibility
                """
                self.points = np.array(points, dtype=float)
                if self.points.ndim != 2 or self.points.shape[1] != 2:
                        raise ValueError("points must be an array-like shape (N,2)")
                self.N = self.points.shape[0]
                self.grid_shape = grid_shape
                self.spacing = spacing
                self.step_length = float(step_length)
                self.collision_distance = float(collision_distance)
                self.max_attempts = int(max_attempts)
                self.max_delay_steps = int(max_delay_steps)
                self.perturb_scale = float(perturb_scale)
                self.rng = np.random.RandomState(random_seed)

        @staticmethod
        def load_from_csv(path, delimiter=","):
                pts = []
                with open(path, newline="") as f:
                        reader = csv.reader(f, delimiter=delimiter)
                        for row in reader:
                                if not row:
                                        continue
                                x, y = float(row[0]), float(row[1])
                                pts.append((x, y))
                return np.array(pts, dtype=float)

        def _choose_grid_shape(self, N):
                if self.grid_shape is not None:
                        r, c = self.grid_shape
                        if r * c < N:
                                raise ValueError("Provided grid_shape too small for number of points")
                        return (int(r), int(c))
                # auto near-square: choose rows = floor(sqrt(N)), cols = ceil(N/rows)
                rows = int(sqrt(N))
                if rows * rows < N:
                        rows = rows
                if rows < 1:
                        rows = 1
                cols = int(ceil(N / rows))
                # adjust to make compact (increase rows if needed to reduce cols)
                while (rows - 1) * cols >= N and rows > 1:
                        rows -= 1
                        cols = int(ceil(N / rows))
                return (rows, cols)

        def create_grid(self, use_perturb=False):
                """
                Create grid nodes centered on mean of points.
                Returns array shape (M,2) where M >= N (grid nodes)
                """
                r, c = self._choose_grid_shape(self.N)
                M = r * c
                center = self.points.mean(axis=0)
                # determine spacing if not given: base on bounding box and desired grid size
                if self.spacing is None:
                        minxy = self.points.min(axis=0)
                        maxxy = self.points.max(axis=0)
                        range_xy = maxxy - minxy
                        # if all points nearly coincident, default spacing to 1.0
                        if np.allclose(range_xy, 0):
                                spacing = 1.0
                        else:
                                # cover the extent but allow some margin
                                sx = range_xy[0] / max(c - 1, 1)
                                sy = range_xy[1] / max(r - 1, 1)
                                spacing = max(sx, sy)
                                if spacing <= 0:
                                        spacing = 1.0
                else:
                        spacing = float(self.spacing)
                # generate grid coordinates centered at origin then shift by center
                xs = (np.arange(c) - (c - 1) / 2.0) * spacing
                ys = (np.arange(r) - (r - 1) / 2.0) * spacing
                xv, yv = np.meshgrid(xs, ys)
                grid_nodes = np.stack((xv.ravel(), yv.ravel()), axis=1) + center
                if use_perturb:
                        # small random perturbation to help avoid symmetric collisions
                        sigma = spacing * self.perturb_scale
                        grid_nodes = grid_nodes + self.rng.normal(scale=sigma, size=grid_nodes.shape)
                return grid_nodes, spacing

        def _assign(self, points, grid_nodes):
                """
                Use Hungarian algorithm to assign each point to a unique grid node.
                Returns:
                    - indices: array shape (N,) of grid node indices assigned to each point
                """
                if linear_sum_assignment is None:
                        # Fallback greedy assignment (not optimal) if scipy unavailable
                        # Still produce a valid assignment
                        cost = np.linalg.norm(points[:, None, :] - grid_nodes[None, :, :], axis=2)
                        indices = np.full(points.shape[0], -1, dtype=int)
                        assigned = np.zeros(grid_nodes.shape[0], dtype=bool)
                        for i in range(points.shape[0]):
                                j = np.argmin(np.where(assigned[None, :], np.inf, cost[i]))
                                indices[i] = j
                                assigned[j] = True
                        return indices
                cost = np.linalg.norm(points[:, None, :] - grid_nodes[None, :, :], axis=2)
                row_ind, col_ind = linear_sum_assignment(cost)
                # row_ind are 0..N-1 (points), col_ind are assigned grid indices
                # linear_sum_assignment returns rows sorted; map to original order
                indices = np.full(points.shape[0], -1, dtype=int)
                indices[row_ind] = col_ind
                return indices

        def _sample_path(self, start, end):
                vec = end - start
                length = np.linalg.norm(vec)
                if length == 0:
                        return np.array([start.copy()])
                n_steps = max(1, int(ceil(length / self.step_length)))
                t = np.linspace(0.0, 1.0, n_steps + 1)
                pts = (1 - t)[:, None] * start[None, :] + t[:, None] * end[None, :]
                return pts  # shape (n_steps+1, 2)

        def _sample_all_paths(self, assignments, grid_nodes):
                """
                assignments: indices array shape (N,) mapping each agent to grid node index
                returns list of arrays: each array shape (T_i, 2) with sampled positions from start to end
                """
                paths = []
                for i in range(self.N):
                        end = grid_nodes[assignments[i]]
                        start = self.points[i]
                        pts = self._sample_path(start, end)
                        paths.append(pts)
                return paths

        def _trajectories_with_delays(self, paths, delays):
                """
                Given list of paths (arrays of positions) and integer delays per agent (in discrete steps),
                produce a trajectories array shape (N, T, 2) padded by repeating final position when path ends.
                The 'delay' means the agent stays at its start position for 'delay' timesteps before moving.
                """
                # convert paths into step-indexed sequences (each step is an entry in the path array)
                lengths = [p.shape[0] for p in paths]
                total_steps = max([d + L for d, L in zip(delays, lengths)])
                N = self.N
                traj = np.zeros((N, total_steps, 2), dtype=float)
                for i in range(N):
                        delay = delays[i]
                        L = lengths[i]
                        # fill delay with start pos
                        start_pos = paths[i][0]
                        traj[i, :delay, :] = start_pos
                        # fill movement steps
                        traj[i, delay:delay + L, :] = paths[i]
                        # fill remainder with final pos
                        final_pos = paths[i][-1]
                        end_idx = delay + L
                        if end_idx < total_steps:
                                traj[i, end_idx:, :] = final_pos
                return traj

        def _detect_collisions(self, trajectories):
                """
                trajectories: array shape (N, T, 2)
                Returns: list of (t, i, j, dist) collisions found where dist < collision_distance
                """
                N, T, _ = trajectories.shape
                collisions = []
                for t in range(T):
                        pts = trajectories[:, t, :]  # (N,2)
                        # pairwise distances
                        diff = pts[None, :, :] - pts[:, None, :]
                        dists = np.linalg.norm(diff, axis=2)
                        # consider only i<j
                        for i in range(N):
                                for j in range(i + 1, N):
                                        if dists[i, j] < self.collision_distance - 1e-12:
                                                collisions.append((t, i, j, float(dists[i, j])))
                return collisions

        def organize(self):
                """
                Main orchestration:
                - build grid (with optional perturbations in attempts)
                - assign via Hungarian
                - sample paths
                - check collisions and attempt to resolve via perturb+reassign or delays
                Returns dict with keys:
                        "assignments": (N,2) target coords,
                        "indices": (N,) assigned grid index,
                        "trajectories": (N, T, 2),
                        "delays": list length N of delays in steps,
                        "success": bool
                """
                # First create nominal grid (no perturb)
                base_grid, spacing = self.create_grid(use_perturb=False)
                best_result = None

                # Attempt 0: initial optimal assignment
                attempts = 0
                tried_perturbs = 0
                while attempts < self.max_attempts:
                        use_perturb = (attempts > 0)
                        grid_nodes = base_grid.copy()
                        if use_perturb:
                                # perturb grid nodes a little to escape symmetric collisions
                                sigma = spacing * self.perturb_scale
                                grid_nodes = grid_nodes + self.rng.normal(scale=sigma, size=grid_nodes.shape)
                                tried_perturbs += 1

                        indices = self._assign(self.points, grid_nodes)
                        paths = self._sample_all_paths(indices, grid_nodes)
                        delays = [0] * self.N
                        trajectories = self._trajectories_with_delays(paths, delays)
                        collisions = self._detect_collisions(trajectories)
                        if not collisions:
                                # success without delays
                                return {
                                        "assignments": grid_nodes[indices],
                                        "indices": indices,
                                        "trajectories": trajectories,
                                        "delays": delays,
                                        "success": True,
                                }

                        # Try to resolve by staggering delays
                        # Simple heuristic: for each collision in chronological order, delay one of the colliding agents
                        # until the collision disappears, up to max_delay_steps. Iterate until either all collisions resolved or limit reached.
                        delays = [0] * self.N
                        resolved = False
                        for delay_round in range(self.max_delay_steps):
                                trajectories = self._trajectories_with_delays(paths, delays)
                                collisions = self._detect_collisions(trajectories)
                                if not collisions:
                                        resolved = True
                                        break
                                # handle collisions: for each collision, pick one agent to delay by 1 (the one with smaller index alternately)
                                for (t, i, j, d) in collisions:
                                        # pick victim: the one with shorter remaining steps (so delaying affects less time),
                                        # otherwise pick randomly to break symmetry
                                        rem_i = paths[i].shape[0] - max(0, t - delays[i])
                                        rem_j = paths[j].shape[0] - max(0, t - delays[j])
                                        if rem_i <= rem_j:
                                                victim = i
                                        else:
                                                victim = j
                                        # increase delay if below max
                                        if delays[victim] < self.max_delay_steps:
                                                delays[victim] += 1
                                # continue next round to re-evaluate collisions
                        if resolved:
                                trajectories = self._trajectories_with_delays(paths, delays)
                                return {
                                        "assignments": grid_nodes[indices],
                                        "indices": indices,
                                        "trajectories": trajectories,
                                        "delays": delays,
                                        "success": True,
                                }

                        # else try another perturbation + reassignment
                        attempts += 1

                # If exhausted attempts, return the last best failed attempt (first assignment) with success False
                # build final trajectories with minimal delays computed above to return some plan
                # Use last computed indices/paths/delays if present
                final_indices = indices
                final_paths = paths
                final_delays = delays
                final_traj = self._trajectories_with_delays(final_paths, final_delays)
                return {
                        "assignments": grid_nodes[final_indices],
                        "indices": final_indices,
                        "trajectories": final_traj,
                        "delays": final_delays,
                        "success": False,
                }


# If run as a script, quick demo with random points (for manual testing)
if __name__ == "__main__":
        import matplotlib.pyplot as plt

        rng = np.random.RandomState(1)
        pts = rng.randn(80, 2) * 2.0
        org = BetterOrganizer(pts, step_length=0.3, collision_distance=0.6, random_seed=1)
        result = org.organize()
        traj = result["trajectories"]
        N, T, _ = traj.shape
        print("Success:", result["success"])
        print("Delays:", result["delays"])

        # Plot initial and final positions and sample trajectories
        plt.figure(figsize=(6, 6))
        for i in range(N):
                path = traj[i]
                plt.plot(path[:, 0], path[:, 1], "-", alpha=0.7)
                plt.plot(path[0, 0], path[0, 1], "ko")
                plt.plot(path[-1, 0], path[-1, 1], "s", mfc="none")
        plt.axis("equal")
        plt.title(f"Trajectories (success={result['success']})")
        plt.show()