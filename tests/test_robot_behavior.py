import pybullet as p
import pybullet_data
import numpy as np
import math
import sys
import os

# Add the src directory to the Python path to import env_utils, dstar_planner
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from env_utils import setup_sim, simulate_lidar, get_odometry, apply_robot_action
from dstar_planner import astar_search # Import astar_search

def test_lidar_obstacle_detection():
    print("\n--- Running LiDAR Obstacle Detection Test ---")
    # Setup a direct PyBullet simulation for testing
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1./240.)

    # Load a plane
    p.loadURDF("plane.urdf")

    # Place robot at origin, facing positive X
    robot_start_pos = [0, 0, 0.1]
    robot_start_orn = p.getQuaternionFromEuler([0, 0, 0]) # Yaw = 0, facing +X
    robot_id = p.loadURDF("r2d2.urdf", robot_start_pos, robot_start_orn)

    # Place a cube obstacle directly in front of the robot
    # Cube is 1x1x1, so halfExtents are 0.5.
    # If robot is at [0,0,0.1], and cube is at [1.5,0,0.5], its front face is at x=1.0
    # Distance from robot center to cube face = 1.0
    obstacle_pos = [1.5, 0, 0.5]
    cube_id = p.loadURDF("cube.urdf", obstacle_pos, globalScaling=1.0)

    # Simulate LiDAR
    num_rays = 16
    ray_length = 5.0
    distances, hit_points = simulate_lidar(robot_id, num_rays, ray_length, show_lasers=False)

    # Expected distance to the obstacle for the ray pointing directly forward (0 degrees yaw)
    # The rays are spread from 0 to 2*pi. The first ray (index 0) is at 0 degrees relative to robot's yaw.
    # So, distances[0] should correspond to the forward ray.
    # LiDAR origin is at [0.5, 0, 0.3]. Obstacle front face is at x=1.0. Distance = 1.0 - 0.5 = 0.5
    expected_distance_forward_ray = 0.5
    
    # Assert the distance for the forward-pointing ray
    # Use a small tolerance for floating point comparisons
    assert np.isclose(distances[0], expected_distance_forward_ray, atol=0.06),         f"Forward ray distance incorrect. Expected: {expected_distance_forward_ray}, Got: {distances[0]}"
    
    # Assert that other rays (e.g., side rays) are not hitting the cube and report max_range
    # For example, rays at 90 degrees (index num_rays/4) and -90 degrees (index 3*num_rays/4)
    # should not hit the cube if it's only directly in front.
    assert np.isclose(distances[num_rays // 4], ray_length, atol=0.06),         f"Side ray (90 deg) distance incorrect. Expected: {ray_length}, Got: {distances[num_rays // 4]}"
    assert np.isclose(distances[3 * num_rays // 4], ray_length, atol=0.06),         f"Side ray (-90 deg) distance incorrect. Expected: {ray_length}, Got: {distances[3 * num_rays // 4]}"

    print("LiDAR Obstacle Detection Test Passed!")
    p.disconnect()

def test_obstacle_avoidance_astar():
    print("\n--- Running Obstacle Avoidance (A*) Test ---")

    # Define a simple grid map with an obstacle
    # 0: Free, 1: Obstacle
    grid_map = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0]
    ])

    start = (0, 0) # Top-left
    goal = (4, 4)  # Bottom-right

    # Run A* search
    path = astar_search(grid_map, start, goal)

    # Assertions
    assert path is not None, "A* search should find a path."
    assert len(path) > 0, "Path should not be empty."

    print(f"Found path: {path}")

    # Verify that the path does not go through any obstacles
    for x, y in path:
        assert grid_map[y, x] == 0, f"Path goes through an obstacle at ({x}, {y})"

    print("Obstacle Avoidance (A*) Test Passed!")

def test_wall_collision():
    print("\n--- Running Wall Collision Test ---")
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1./240.)

    p.loadURDF("plane.urdf")

    # Create a wall
    wall_half_extents = [0.1, 2.0, 1.0] # Thin in X, long in Y, tall in Z
    wall_pos = [1.0, 0, wall_half_extents[2]] # Wall at x=1.0
    wall_collision_shape_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=wall_half_extents)
    wall_id = p.createMultiBody(baseMass=0,
                                baseCollisionShapeIndex=wall_collision_shape_id,
                                basePosition=wall_pos)

    # Place robot near the wall, facing it
    robot_start_pos = [0.5, 0, 0.1] # Robot at x=0.5
    robot_start_orn = p.getQuaternionFromEuler([0, 0, 0]) # Facing +X
    robot_id = p.loadURDF("r2d2.urdf", robot_start_pos, robot_start_orn)

    # Apply forward motion for a few steps
    num_simulation_steps = 100
    for _ in range(num_simulation_steps):
        apply_robot_action(robot_id, 0) # Move forward
        p.stepSimulation()

    # Get robot's final position
    final_pose = get_odometry(robot_id)
    final_x = final_pose[0]

    # Calculate the wall's front face x-coordinate
    # Wall is at x=1.0, half_extents[0] = 0.1. So its front face is at 1.0 - 0.1 = 0.9
    wall_front_x = wall_pos[0] - wall_half_extents[0]

    # Assert that the robot's x-position does not exceed the wall's front face
    # Allow a small tolerance for physics engine inaccuracies
    # The robot's "front" is not a single point, so we need to consider its size.
    # R2D2's approximate radius is around 0.5-0.6 units.
    # So, robot's center should not go beyond wall_front_x + robot_radius
    robot_approx_radius = 0.6 # Estimate based on R2D2 model
    assert final_x < (wall_front_x + robot_approx_radius + 0.05), \
        f"Robot penetrated the wall! Final X: {final_x}, Wall Front X: {wall_front_x}"

    print("Wall Collision Test Passed!")
    p.disconnect()

def reset_robot_pose(robot_id, pos, orn):
    """Helper function to reset robot's base position and orientation."""
    p.resetBasePositionAndOrientation(robot_id, pos, orn)
    p.resetBaseVelocity(robot_id, [0,0,0], [0,0,0]) # Also reset velocity

def test_motion_and_direction():
    print("\n--- Running Motion and Direction Test ---")
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1./240.)

    p.loadURDF("plane.urdf")

    initial_pos = [0, 0, 0.1]
    initial_orn = p.getQuaternionFromEuler([0, 0, 0]) # Yaw = 0, facing +X
    robot_id = p.loadURDF("r2d2.urdf", initial_pos, initial_orn)

    # --- Test Move Forward (action 0) ---
    print("Testing Move Forward...")
    reset_robot_pose(robot_id, initial_pos, initial_orn)
    initial_pose = get_odometry(robot_id)
    
    num_steps_forward = 50
    for _ in range(num_steps_forward):
        apply_robot_action(robot_id, 0) # Move forward
        p.stepSimulation()
    
    final_pose = get_odometry(robot_id)
    assert final_pose[0] > initial_pose[0], f"Robot did not move forward. Initial X: {initial_pose[0]}, Final X: {final_pose[0]}"
    assert np.isclose(final_pose[1], initial_pose[1], atol=0.1), f"Robot moved sideways. Initial Y: {initial_pose[1]}, Final Y: {final_pose[1]}"
    assert np.isclose(final_pose[2], initial_pose[2], atol=0.1), f"Robot changed yaw. Initial Yaw: {initial_pose[2]}, Final Yaw: {final_pose[2]}"
    print("Move Forward Test Passed.")

    # --- Test Turn Left (action 1) ---
    print("Testing Turn Left...")
    reset_robot_pose(robot_id, initial_pos, initial_orn)
    initial_pose = get_odometry(robot_id)
    
    num_steps_turn = 20
    for _ in range(num_steps_turn):
        apply_robot_action(robot_id, 1) # Turn left
        p.stepSimulation()
    
    final_pose = get_odometry(robot_id)
    # Yaw should increase. Handle angle wrapping.
    initial_yaw_norm = (initial_pose[2] + 2 * np.pi) % (2 * np.pi)
    final_yaw_norm = (final_pose[2] + 2 * np.pi) % (2 * np.pi)
    
    # Check if yaw increased, considering wrap around (e.g., from 350 to 10 degrees)
    yaw_diff = final_yaw_norm - initial_yaw_norm
    if yaw_diff < -np.pi: yaw_diff += 2 * np.pi
    if yaw_diff > np.pi: yaw_diff -= 2 * np.pi

    assert yaw_diff > 0.1, f"Robot did not turn left. Yaw Diff: {yaw_diff}"
    assert np.isclose(final_pose[0], initial_pose[0], atol=0.1), f"Robot moved in X. Initial X: {initial_pose[0]}, Final X: {final_pose[0]}"
    assert np.isclose(final_pose[1], initial_pose[1], atol=0.1), f"Robot moved in Y. Initial Y: {initial_pose[1]}, Final Y: {final_pose[1]}"
    print("Turn Left Test Passed.")

    # --- Test Turn Right (action 2) ---
    print("Testing Turn Right...")
    reset_robot_pose(robot_id, initial_pos, initial_orn)
    initial_pose = get_odometry(robot_id)
    
    num_steps_turn = 20
    for _ in range(num_steps_turn):
        apply_robot_action(robot_id, 2) # Turn right
        p.stepSimulation()
    
    final_pose = get_odometry(robot_id)
    # Yaw should decrease. Handle angle wrapping.
    initial_yaw_norm = (initial_pose[2] + 2 * np.pi) % (2 * np.pi)
    final_yaw_norm = (final_pose[2] + 2 * np.pi) % (2 * np.pi)
    
    yaw_diff = final_yaw_norm - initial_yaw_norm
    if yaw_diff < -np.pi: yaw_diff += 2 * np.pi
    if yaw_diff > np.pi: yaw_diff -= 2 * np.pi

    assert yaw_diff < -0.1, f"Robot did not turn right. Yaw Diff: {yaw_diff}"
    assert np.isclose(final_pose[0], initial_pose[0], atol=0.1), f"Robot moved in X. Initial X: {initial_pose[0]}, Final X: {final_pose[0]}"
    assert np.isclose(final_pose[1], initial_pose[1], atol=0.1), f"Robot moved in Y. Initial Y: {initial_pose[1]}, Final Y: {final_pose[1]}"
    print("Turn Right Test Passed.")

    # --- Test Stop (action 3) ---
    print("Testing Stop...")
    reset_robot_pose(robot_id, initial_pos, initial_orn)
    
    # Apply forward motion for a few steps to give it velocity
    for _ in range(10):
        apply_robot_action(robot_id, 0)
        p.stepSimulation()
    
    # Record pose after initial movement
    pose_before_stop = get_odometry(robot_id)

    # Apply stop action
    apply_robot_action(robot_id, 3) # Stop
    
    num_steps_stop = 50
    for _ in range(num_steps_stop):
        p.stepSimulation() # Continue stepping to ensure it stays stopped
    
    final_pose_after_stop = get_odometry(robot_id)
    
    assert np.isclose(final_pose_after_stop[0], pose_before_stop[0], atol=0.05), f"Robot moved in X after stop. Before: {pose_before_stop[0]}, After: {final_pose_after_stop[0]}"
    assert np.isclose(final_pose_after_stop[1], pose_before_stop[1], atol=0.05), f"Robot moved in Y after stop. Before: {pose_before_stop[1]}, After: {final_pose_after_stop[1]}"
    assert np.isclose(final_pose_after_stop[2], pose_before_stop[2], atol=0.05), f"Robot rotated after stop. Before: {pose_before_stop[2]}, After: {final_pose_after_stop[2]}"
    print("Stop Test Passed.")

    print("Motion and Direction Test Passed!")
    p.disconnect()

if __name__ == "__main__":
    test_lidar_obstacle_detection()
    test_obstacle_avoidance_astar()
    test_wall_collision()
    test_motion_and_direction() # Add the new test to be run