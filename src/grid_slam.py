import numpy as np
from numba import jit

@jit(nopython=True)
def _update_map_jit(grid_map, pose, scan, log_odds_free, log_odds_occupied):
    map_shape_x = grid_map.shape[0]
    map_shape_y = grid_map.shape[1]

    # For each lidar beam, mark free cells and end cell as occupied
    for i, dist in enumerate(scan):
        angle = pose[2] + i * (2 * np.pi / len(scan))
        
        # Trace ray from robot to hit point
        for r_idx in range(0, int(dist*20) + 1):
            r = r_idx * 0.05 # Step along the ray
            x = int(pose[0] + r * np.cos(angle))
            y = int(pose[1] + r * np.sin(angle))
            
            if 0 <= x < map_shape_x and 0 <= y < map_shape_y:
                if r < dist - 0.2:  # Free space
                    grid_map[x, y] += log_odds_free
                else:  # Occupied space
                    grid_map[x, y] += log_odds_occupied

class GridSLAM:
    def __init__(self, map_size):
        self.map = np.zeros(map_size)
        self.log_odds_free = -0.4
        self.log_odds_occupied = 0.8

        # Mark boundaries as obstacles
        rows, cols = map_size
        self.map[0, :] = 1.0 # A large log-odds value for occupied
        self.map[rows - 1, :] = 1.0
        self.map[:, 0] = 1.0
        self.map[:, cols - 1] = 1.0

    def update(self, pose, scan):
        # Ensure pose is within map bounds before updating
        if not (0 <= pose[0] < self.map.shape[0] and 0 <= pose[1] < self.map.shape[1]):
            return

        _update_map_jit(self.map, pose, scan, self.log_odds_free, self.log_odds_occupied)
        # Clip log-odds values to prevent overflow and overconfidence
        self.map = np.clip(self.map, -10.0, 10.0) # Example values, can be tuned

        # Debug print: Show a small section of the map around the robot's pose
        map_x, map_y = self.map.shape
        px, py = int(pose[0]), int(pose[1])
        window_size = 5
        slice_x_min = max(0, px - window_size // 2)
        slice_x_max = min(map_x, px + window_size // 2 + 1)
        slice_y_min = max(0, py - window_size // 2)
        slice_y_max = min(map_y, py + window_size // 2 + 1)
        print(f"DEBUG: GridSLAM Map around pose ({px},{py}):\n{self.get_map()[slice_x_min:slice_x_max, slice_y_min:slice_y_max]}")

    def get_map(self):
        # Convert log-odds map to probability map
        prob_map = 1.0 - 1.0 / (1.0 + np.exp(self.map))
        return prob_map
