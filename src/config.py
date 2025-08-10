# Simulation and algorithm parameters
MOTION_MODE = 'ASTAR'  # Options: 'RL' for exploration, 'ASTAR' for path following
MAP_SIZE = (300, 300)
LIDAR_RAYS = 16 # Reduced for performance
LIDAR_RANGE = 5.0
PARTICLE_COUNT = 500 # Increased for localization stability debugging
STATE_SPACE_SIZE = 4
RL_ACTIONS = 4
