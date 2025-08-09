import pybullet as p
import pybullet_data
import numpy as np
import math

def setup_sim(gui=True):
    """Setup PyBullet simulation with proper physics"""
    p.connect(p.GUI if gui else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1./240.)
    
    # Load ground plane
    plane_id = p.loadURDF("plane.urdf")
    
    # Robot start position
    robot_start_pos = [2, 2, 0.1]  # Slightly above ground
    robot_id = p.loadURDF("r2d2.urdf", robot_start_pos)
    
    # Obstacle positions
    obstacle_positions = [
        (8, 8), (12, 12), (12, 5),
        (5, 5), (5, 10), (10, 5)
    ]
    for i, (x, y) in enumerate(obstacle_positions):
        cube_id = p.loadURDF("cube.urdf", [x, y, 0.5], globalScaling=1.2)
        print(f"Placed obstacle {i+1} at ({x}, {y})")
    
    # Create bounding region (15x15 area)
    env_size = 15
    wall_positions = [
        [env_size/2, 0, 0.5], [env_size/2, env_size, 0.5],
        [0, env_size/2, 0.5], [env_size, env_size/2, 0.5]
    ]
    wall_half_extents = [
        [env_size/2, 0.1, 0.5], [env_size/2, 0.1, 0.5],
        [0.1, env_size/2, 0.5], [0.1, env_size/2, 0.5]
    ]

    for i in range(4):
        p.createMultiBody(baseMass=0,
                          baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=wall_half_extents[i]),
                          basePosition=wall_positions[i])
    print("Created 15x15 bounding region.")

    # Add start and goal markers
    goal_pos = [13, 13, 0.1]
    start_marker = p.createVisualShape(p.GEOM_SPHERE, radius=0.3, rgbaColor=[0, 1, 0, 0.8])
    goal_marker = p.createVisualShape(p.GEOM_SPHERE, radius=0.3, rgbaColor=[1, 0, 0, 0.8])

    p.createMultiBody(baseVisualShapeIndex=start_marker, basePosition=robot_start_pos)
    p.createMultiBody(baseVisualShapeIndex=goal_marker, basePosition=goal_pos)
    print("Added start and goal markers.")

    print(f"Robot loaded with ID: {robot_id}")
    return robot_id

def get_odometry(robot_id):
    """Get robot's current pose"""
    pos, orn = p.getBasePositionAndOrientation(robot_id)
    euler = p.getEulerFromQuaternion(orn)
    return np.array([pos[0], pos[1], euler[2]])  # x, y, yaw

def apply_robot_action(robot_id, action):
    """Apply movement commands to robot - FIXED VERSION"""
    # Get number of joints to understand robot structure
    num_joints = p.getNumJoints(robot_id)
    
    # For r2d2, we can either use resetBaseVelocity or control wheel joints if they exist
    if action == 0:  # Move forward
        p.resetBaseVelocity(robot_id, linearVelocity=[5.0, 0, 0])
        print("Action: Move forward")
    elif action == 1:  # Turn left
        p.resetBaseVelocity(robot_id, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 6.0])
        print("Action: Turn left")
    elif action == 2:  # Turn right
        p.resetBaseVelocity(robot_id, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, -6.0])
        print("Action: Turn right")
    else:  # Stop
        p.resetBaseVelocity(robot_id, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])
        print("Action: Stop")

def simulate_lidar(robot_id, num_rays=16, ray_length=5.0, show_lasers=False):
    """Simulate LiDAR sensor using raycast"""
    pos, orn = p.getBasePositionAndOrientation(robot_id)
    euler = p.getEulerFromQuaternion(orn)
    yaw = euler[2]
    
    rays_from = []
    rays_to = []
    
    for i in range(num_rays):
        angle = yaw + (i * 2 * math.pi / num_rays)
        start_pos = [pos[0], pos[1], pos[2] + 0.2]  # Slightly above robot
        end_pos = [
            pos[0] + ray_length * math.cos(angle),
            pos[1] + ray_length * math.sin(angle),
            pos[2] + 0.2
        ]
        rays_from.append(start_pos)
        rays_to.append(end_pos)
    
    # Perform raycast
    ray_results = p.rayTestBatch(rays_from, rays_to)
    distances = []
    
    if show_lasers:
        p.removeAllUserDebugItems()

    for i, result in enumerate(ray_results):
        hit_fraction = result[2]
        hit_position = result[3]
        
        if hit_fraction < 1.0:
            distance = hit_fraction * ray_length
            if show_lasers:
                p.addUserDebugLine(rays_from[i], hit_position, [1, 0, 0]) # Red for hit
        else:
            distance = ray_length
            if show_lasers:
                p.addUserDebugLine(rays_from[i], rays_to[i], [0, 1, 0]) # Green for no hit

        distances.append(distance)
    
    return np.array(distances)
