import time
import numpy as np
from config import MAP_SIZE, LIDAR_RAYS, LIDAR_RANGE, PARTICLE_COUNT, RL_ACTIONS, MOTION_MODE, STATE_SPACE_SIZE
from env_utils import setup_sim, get_odometry, apply_robot_action, simulate_lidar
from markov_localization import MarkovLocalization
from grid_slam import GridSLAM
from dstar_planner import astar_search
from rl_agent import RLAgent
from pl_icp import pl_icp_correction
import pybullet as p

def run():
    print("Setting up PyBullet simulation...")
    robot_id = setup_sim(gui=True)
    
    # Initialize algorithms
    mcl = MarkovLocalization(PARTICLE_COUNT, MAP_SIZE)
    slam = GridSLAM(MAP_SIZE)
    rl_agent = RLAgent(RL_ACTIONS)
    
    # Goal coordinates for path planning
    goal_coords = (13, 13)
    
    print("="*50)
    print("🤖 ASSIGNMENT 2: Multi-Sensor Robot Navigation")
    print(f" MOTION MODE: {MOTION_MODE}")
    print("="*50)
    print("✓ Markov Localization (Particle Filter)")
    print("✓ Grid-based SLAM")
    print("✓ D* Path Planning")
    print("✓ RL-based Exploration")
    print("✓ PL-ICP Pose Correction")
    print("="*50)
    print("Simulation running! Press Ctrl+C to quit.\n")
    
    start_time = time.time()
    step_count = 0
    prev_pose = get_odometry(robot_id)
    start_pose = prev_pose
    total_distance = 0
    prev_scan = None
    prev_scan_points = None
    path = []
    action = 3 # Default action is STOP
    turn_attempts = 0 # New variable to track consecutive turns when no path is found
    exploration_mode = False
    exploration_step_count = 0

    try:
        while True:
            # Get current robot state
            current_pose = get_odometry(robot_id)
            scan, scan_points = simulate_lidar(robot_id, LIDAR_RAYS, LIDAR_RANGE, show_lasers=True)
            
            # Calculate motion since last step (odometry)
            if prev_pose is not None:
                delta_pose = current_pose - prev_pose
                delta_pose[2] = (delta_pose[2] + np.pi) % (2 * np.pi) - np.pi
                distance_moved = np.linalg.norm(delta_pose[:2])
                total_distance += distance_moved
            else:
                delta_pose = np.zeros(3)
            
            grid_map = slam.get_map() 
            
            # 1. LOCALIZATION (MCL)
            mcl.motion_update(delta_pose)
            mcl.measurement_update(scan, grid_map)
            est_pose = mcl.get_estimated_pose()
            
            # Clip estimated pose to be within map boundaries
            est_pose[0] = np.clip(est_pose[0], 0, MAP_SIZE[0]-1)
            est_pose[1] = np.clip(est_pose[1], 0, MAP_SIZE[1]-1)

            # (Optional) Refine pose estimate with PL-ICP
            if prev_scan_points is not None:
                est_pose = pl_icp_correction(scan_points, prev_scan_points, est_pose)
            
            # 2. MAPPING (SLAM)
            slam.update(est_pose, scan)

            # 3. VISUALIZE PARTICLES
            particles = mcl.get_particles()
            if particles:
                positions = [[p.x, p.y, 0.1] for p in particles]
                weights = [p.weight for p in particles]
                max_w = max(weights) if max(weights) > 0 else 1.0
                colors = [[1 - (w/max_w), (w/max_w), 0] for w in weights]
                p.addUserDebugPoints(positions, colors, pointSize=5)
            
            # 4. MOTION PLANNING (RL or A*)
            if MOTION_MODE == 'RL':
                # RL Agent: Choose action based on exploration reward
                num_unknown = np.sum(grid_map == 0)
                reward = num_unknown / (MAP_SIZE[0] * MAP_SIZE[1])
                current_state = rl_agent._discretize_state(num_unknown, MAP_SIZE[0] * MAP_SIZE[1])
                action = rl_agent.select_action(current_state)
                
                next_num_unknown = np.sum(slam.get_map() == 0)
                next_state = rl_agent._discretize_state(next_num_unknown, MAP_SIZE[0] * MAP_SIZE[1])
                rl_agent.update(current_state, action, reward, next_state)

            elif MOTION_MODE == 'ASTAR':
                # A* Path Following
                if not path or step_count % 60 == 0: # Re-plan every 2 seconds or if no path
                    try:
                        current_grid_pos = (int(est_pose[0]), int(est_pose[1]))
                        print(f"DEBUG: Current Grid Pos: {current_grid_pos}, Goal Coords: {goal_coords}")
                        print(f"DEBUG: Grid Map Shape: {grid_map.shape}")
                        
                        # Print a small section of the grid map around start and goal
                        # Ensure coordinates are within bounds before slicing
                        map_x, map_y = grid_map.shape
                        
                        start_x, start_y = current_grid_pos
                        goal_x, goal_y = goal_coords

                        # Define a small window around the start and goal
                        window_size = 5
                        
                        # For start
                        start_slice_x_min = max(0, start_x - window_size // 2)
                        start_slice_x_max = min(map_x, start_x + window_size // 2 + 1)
                        start_slice_y_min = max(0, start_y - window_size // 2)
                        start_slice_y_max = min(map_y, start_y + window_size // 2 + 1)
                        
                        print(f"DEBUG: Grid around start ({start_x},{start_y}):\n{grid_map[start_slice_x_min:start_slice_x_max, start_slice_y_min:start_slice_y_max]}")

                        # For goal
                        goal_slice_x_min = max(0, goal_x - window_size // 2)
                        goal_slice_x_max = min(map_x, goal_x + window_size // 2 + 1)
                        goal_slice_y_min = max(0, goal_y - window_size // 2)
                        goal_slice_y_max = min(map_y, goal_y + window_size // 2 + 1)

                        print(f"DEBUG: Grid around goal ({goal_x},{goal_y}):\n{grid_map[goal_slice_x_min:goal_slice_x_max, goal_slice_y_min:goal_slice_y_max]}")

                        path = astar_search(grid_map, current_grid_pos, goal_coords)
                        if len(path) > 1:
                            print(f"📍 Path found: {path[:3]}... (Total length: {len(path)})")
                        else:
                            print(f"⚠️ No path found or path too short. Path: {path}")
                            path = []
                    except Exception as e:
                        print(f"⚠️ Path planning failed: {e}")
                        path = []

                if path:
                    target_waypoint = path[0]
                    dist_to_target = np.linalg.norm(np.array([est_pose[0], est_pose[1]]) - np.array(target_waypoint))

                    delta_y = target_waypoint[1] - est_pose[1]
                    delta_x = target_waypoint[0] - est_pose[0]
                    desired_angle = np.arctan2(delta_y, delta_x)
                    angle_diff = (desired_angle - est_pose[2] + np.pi) % (2 * np.pi) - np.pi

                    # Prioritize rotation if angle is significantly off
                    if abs(angle_diff) > 0.5: # Smaller threshold for rotation
                        action = 1 if angle_diff > 0 else 2 # Turn right or left
                    elif dist_to_target < 0.5: # If very close to waypoint, pop it
                        path.pop(0)
                        if not path:
                            action = 3 # Stop, path complete
                            time_taken = time.time() - start_time
                            print("\n" + "="*50)
                            print(f"🏁 GOAL REACHED! 🏁")
                            print(f"   - Time Taken: {time_taken:.2f} seconds")
                            print(f"   - Distance Traveled: {total_distance:.2f} meters")
                            print("="*50 + "\n")
                        else:
                            target_waypoint = path[0] # Update target
                            action = 0 # Move forward to next waypoint
                    else:
                        action = 0 # Move forward

                else:
                    # No path found, enter exploration mode
                    if not exploration_mode:
                        exploration_mode = True
                        exploration_step_count = 0
                        print("DEBUG: No path found. Entering exploration mode.")

                    # Exploration logic
                    if exploration_mode:
                        if exploration_step_count < 10: # Move forward for 10 steps
                            action = 0 # Move forward
                            print(f"DEBUG: Exploration: Moving forward (step {exploration_step_count + 1}).")
                        elif exploration_step_count < 15: # Then turn left for 5 steps
                            action = 1 # Turn left
                            print(f"DEBUG: Exploration: Turning left (step {exploration_step_count + 1}).")
                        else:
                            # Exit exploration mode after a cycle
                            exploration_mode = False
                            exploration_step_count = 0
                            action = 3 # Stop for a moment before re-planning
                            print("DEBUG: Exploration cycle complete. Stopping.")
                        exploration_step_count += 1

                # Reset exploration mode if a path is found
                if path:
                    exploration_mode = False
                    exploration_step_count = 0

            apply_robot_action(robot_id, action)

            # Update state for next iteration
            prev_pose = current_pose.copy()
            prev_scan = scan.copy()
            prev_scan_points = scan_points.copy()
            
            # Print status every 10 steps
            if step_count % 10 == 0:
                print(f"Step {step_count:4d} | Pose: ({est_pose[0]:5.2f}, {est_pose[1]:5.2f}, {est_pose[2]:5.2f}) | "
                      f"Mode: {MOTION_MODE} | Action: {action}")
            
            # Step physics simulation and sleep
            step_count += 1
            p.stepSimulation()
            time.sleep(1/60)
            
    except KeyboardInterrupt:
        print("\n🛑 Simulation stopped by user.")
        print("📊 Final Statistics:")
        print(f"   - Total steps: {step_count}")
        print(f"   - Final pose: ({est_pose[0]:.2f}, {est_pose[1]:.2f}, {est_pose[2]:.2f})")
        print(f"   - Map coverage: {np.sum(grid_map > 0.1)}/{MAP_SIZE[0]*MAP_SIZE[1]} cells")
        p.disconnect()

if __name__ == "__main__":
    run()