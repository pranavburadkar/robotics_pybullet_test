import numpy as np

class GridSLAM:
    def __init__(self, map_size):
        self.map = np.zeros(map_size)

    def update(self, pose, scan):
        # For each lidar beam, mark free cells and end cell as occupied
        for i, dist in enumerate(scan):
            angle = pose[2] + i * (2 * np.pi / len(scan))
            
            # Trace ray from robot to hit point
            for r in np.linspace(0, dist, int(dist*10) + 1):
                x = int(pose[0] + r * np.cos(angle))
                y = int(pose[1] + r * np.sin(angle))
                
                if 0 <= x < self.map.shape[0] and 0 <= y < self.map.shape[1]:
                    if r < dist - 0.2:  # Free space
                        self.map[x, y] = max(self.map[x, y] - 0.1, 0)
                    else:  # Occupied space
                        self.map[x, y] = min(self.map[x, y] + 0.3, 1)

    def get_map(self):
        return self.map
