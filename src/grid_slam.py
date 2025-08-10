import numpy as np

class GridSLAM:
    def __init__(self, map_size):
        self.map = np.zeros(map_size)
        # Mark boundaries as obstacles
        rows, cols = map_size
        self.map[0, :] = 1.0  # Top boundary
        self.map[rows - 1, :] = 1.0  # Bottom boundary
        self.map[:, 0] = 1.0  # Left boundary
        self.map[:, cols - 1] = 1.0  # Right boundary

    def update(self, pose, scan):
        # Ensure pose is within map bounds before updating
        if not (0 <= pose[0] < self.map.shape[0] and 0 <= pose[1] < self.map.shape[1]):
            return

        # For each lidar beam, mark free cells and end cell as occupied
        for i, dist in enumerate(scan):
            angle = pose[2] + i * (2 * np.pi / len(scan))
            
            # Trace ray from robot to hit point
            for r in np.linspace(0, dist, int(dist*20) + 1):
                x = int(pose[0] + r * np.cos(angle))
                y = int(pose[1] + r * np.sin(angle))
                
                if 0 <= x < self.map.shape[0] and 0 <= y < self.map.shape[1]:
                    if r < dist - 0.2:  # Free space
                        self.map[x, y] = max(self.map[x, y] - 0.1, 0)
                    else:  # Occupied space
                        self.map[x, y] = min(self.map[x, y] + 0.3, 1)
                        # Inflate neighbors
                        for dx in [-1, 0, 1]:
                            for dy in [-1, 0, 1]:
                                if dx == 0 and dy == 0: continue
                                nx, ny = x + dx, y + dy
                                if 0 <= nx < self.map.shape[0] and 0 <= ny < self.map.shape[1]:
                                    self.map[nx, ny] = min(self.map[nx, ny] + 0.2, 1) # Inflate by a smaller amount

    def get_map(self):
        return self.map
