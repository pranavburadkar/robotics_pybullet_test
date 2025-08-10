import numpy as np

class GridSLAM:
    def __init__(self, map_size):
        self.map = np.zeros(map_size)
        self.log_odds_free = -0.4
        self.log_odds_occupied = 0.8

        # Mark boundaries as obstacles
        rows, cols = map_size
        self.map[0, :] = 100 # A large log-odds value for occupied
        self.map[rows - 1, :] = 100
        self.map[:, 0] = 100
        self.map[:, cols - 1] = 100

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
                        self.map[x, y] += self.log_odds_free
                    else:  # Occupied space
                        self.map[x, y] += self.log_odds_occupied

    def get_map(self):
        # Convert log-odds map to probability map
        prob_map = 1.0 - 1.0 / (1.0 + np.exp(self.map))
        return prob_map
