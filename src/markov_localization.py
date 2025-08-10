import numpy as np
from numba import jit

class Particle:
    """A simple data structure for a particle in the filter."""
    def __init__(self, x, y, theta, weight=1.0):
        self.x = x
        self.y = y
        self.theta = theta
        self.weight = weight

@jit(nopython=True)
def _calculate_particle_weight_jit(particle_x, particle_y, particle_theta, lidar_scan, grid_map):
    """Calculate the likelihood of a measurement given a particle's pose."""
    weight = 1.0
    num_rays = len(lidar_scan)
    for i, measured_dist in enumerate(lidar_scan):
        angle = particle_theta + (i * 2 * np.pi / num_rays)
        expected_dist = _raycast_on_grid_jit(particle_x, particle_y, angle, grid_map, 5.0) # max_range from config
        
        # Compare expected vs. measured distance using a Gaussian model
        error = measured_dist - expected_dist
        # The smaller the error, the higher the probability (and weight)
        prob = np.exp(-(error**2) / (2 * 5.0**2))  # Sensor noise variance
        weight *= prob
    return weight

@jit(nopython=True)
def _raycast_on_grid_jit(particle_x, particle_y, angle, grid_map, max_range):
    """Simulate a Lidar ray on the occupancy grid to find expected distance."""
    map_shape_x = grid_map.shape[0]
    map_shape_y = grid_map.shape[1]
    for r_idx in range(0, int(max_range / 0.05)): # Step along the ray
        r = r_idx * 0.05
        x = int(particle_x + r * np.cos(angle))
        y = int(particle_y + r * np.sin(angle))

        if not (0 <= x < map_shape_x and 0 <= y < map_shape_y):
            return max_range  # Ray went out of map bounds
        if grid_map[x, y] > 0.5:  # Obstacle threshold
            return r  # Ray hit an obstacle
    return max_range  # No obstacle hit within range

class MarkovLocalization:
    """
    Implements a particle filter for robot localization (Monte Carlo Localization).
    """
    def __init__(self, particle_count, map_shape):
        self.particle_count = particle_count
        self.map_shape = map_shape
        self.particles = self._initialize_particles()

    def _initialize_particles(self):
        """Create a set of particles uniformly distributed across the map."""
        particles = []
        for _ in range(self.particle_count):
            x = np.random.uniform(0, self.map_shape[0])
            y = np.random.uniform(0, self.map_shape[1])
            theta = np.random.uniform(0, 2 * np.pi)
            particles.append(Particle(x, y, theta))
        return particles

    def motion_update(self, delta_pose, noise=np.array([0.1, 0.1, 0.05])):
        """
        Update each particle's pose based on a motion model and add noise.
        This is a simplified motion model, not a true EKF.
        """
        for p in self.particles:
            p.x += delta_pose[0] + np.random.normal(0, noise[0])
            p.y += delta_pose[1] + np.random.normal(0, noise[1])
            p.theta += delta_pose[2] + np.random.normal(0, noise[2])
            # Keep particles within map bounds
            p.x = np.clip(p.x, 0, self.map_shape[0]-1)
            p.y = np.clip(p.y, 0, self.map_shape[1]-1)
            p.theta = p.theta % (2 * np.pi)

    def measurement_update(self, lidar_scan, grid_map):
        """
        Update particle weights based on the sensor measurement (LIDAR scan)
        and then resample the particles.
        """
        for p in self.particles:
            p.weight = _calculate_particle_weight_jit(p.x, p.y, p.theta, lidar_scan, grid_map)

        # Normalize weights to form a probability distribution
        total_weight = sum(p.weight for p in self.particles)
        if total_weight < 1e-9:  # Avoid division by zero
            # If all weights are zero, re-initialize them uniformly
            for p in self.particles:
                p.weight = 1.0 / self.particle_count
        else:
            for p in self.particles:
                p.weight /= total_weight

        self._resample_particles()

    def _resample_particles(self):
        """Resample particles based on their weights (low variance sampling)."""
        new_particles = []
        weights = np.array([p.weight for p in self.particles])
        
        # Use numpy's random.choice for efficient weighted resampling
        indices = np.random.choice(
            np.arange(self.particle_count),
            size=self.particle_count,
            p=weights
        )
        for i in indices:
            p = self.particles[i]
            new_particles.append(Particle(p.x, p.y, p.theta, 1.0/self.particle_count))
        self.particles = new_particles

    def get_estimated_pose(self):
        """Calculate the estimated pose as the mean of the particle set."""
        mean_x = np.mean([p.x for p in self.particles])
        mean_y = np.mean([p.y for p in self.particles])
        # Averaging angles requires using their vector components
        mean_theta_x = np.mean([np.cos(p.theta) for p in self.particles])
        mean_theta_y = np.mean([np.sin(p.theta) for p in self.particles])
        mean_theta = np.arctan2(mean_theta_y, mean_theta_x)
        return np.array([mean_x, mean_y, mean_theta])

    def get_particles(self):
        """Return the current set of particles."""
        return self.particles
