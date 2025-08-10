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

def find_unexplored_target(grid_map, current_pose_grid):
    """
    Finds a target point in an unexplored region of the grid map, prioritizing frontier cells.
    """
    map_shape_x, map_shape_y = grid_map.shape
    unknown_cells = np.argwhere((grid_map > 0.4) & (grid_map < 0.6)) # Cells close to 0.5 probability

    if len(unknown_cells) == 0:
        return None # No unexplored cells found

    frontier_cells = []
    for ux, uy in unknown_cells:
        # Check neighbors for known and free cells
        is_frontier = False
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue # Skip self
                nx, ny = ux + dx, uy + dy
                if 0 <= nx < map_shape_x and 0 <= ny < map_shape_y:
                    if grid_map[nx, ny] < 0.4: # Known and free cell
                        is_frontier = True
                        break
            if is_frontier:
                break
        if is_frontier:
            frontier_cells.append((ux, uy))

    target_candidates = []
    if len(frontier_cells) > 0:
        target_candidates = frontier_cells
    else:
        target_candidates = unknown_cells # Fallback to any unknown cell if no frontiers

    valid_target_candidates = []
    for cell in target_candidates:
        dist = np.linalg.norm(np.array(cell) - current_pose_grid[:2])
        if dist > 10: # Only consider cells at least 10 grid units away
            valid_target_candidates.append(cell)
    
    if len(valid_target_candidates) == 0:
        # If all valid candidates are too close, just pick a random one from all target_candidates
        if len(target_candidates) > 0:
            target_cell_idx = np.random.choice(len(target_candidates))
            target_cell = target_candidates[target_cell_idx]
        else:
            return None # No suitable target found
    else:
        # Pick a random valid target candidate
        target_cell_idx = np.random.choice(len(valid_target_candidates))
        target_cell = valid_target_candidates[target_cell_idx]

    return (int(target_cell[0]), int(target_cell[1]))

def find_nearest_free_cell(grid_map, target_x, target_y, search_radius=5):
    """
    Finds the nearest free cell (value < 0.5) to the target coordinates within a search radius.
    """
    map_x, map_y = grid_map.shape
    
    # Check the target cell first
    if 0 <= target_x < map_x and 0 <= target_y < map_y and grid_map[target_y, target_x] < 0.5:
        return (target_x, target_y)

    # Expand search outwards in a spiral or square pattern
    for r in range(1, search_radius + 1):
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                if abs(dx) != r and abs(dy) != r: # Only check cells on the perimeter of the square
                    continue

                nx, ny = target_x + dx, target_y + dy
                if 0 <= nx < map_x and 0 <= ny < map_y and grid_map[ny, nx] < 0.5:
                    return (nx, ny)
    return None # No free cell found within radius

def run():
    print("Setting up PyBullet simulation...")
    robot_id = setup_sim(gui=True)
    
    # Initialize algorithms
    mcl = MarkovLocalization(PARTICLE_COUNT, MAP_SIZE)
    slam = GridSLAM(MAP_SIZE)
    rl_agent = RLAgent(RL_ACTIONS)
    
    # Goal coordinates for path planning
    goal_coords_meters = (13, 13)
    goal_coords = (int(goal_coords_meters[0] / 0.05), int(goal_coords_meters[1] / 0.05))
    
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
    
    exploration_mode = False
    exploration_step_count = 0
    collision_avoidance_mode = False # New state variable

    try:
        while True:
            # Get current robot state
            current_pose = get_odometry(robot_id)
            scan, scan_points = simulate_lidar(robot_id, LIDAR_RAYS, LIDAR_RANGE, show_lasers=True)
            
            # Collision Avoidance: Proactive and Smooth
            collision_threshold = 1.0 # meters (trigger for decisive avoidance)
            warning_threshold = 3.0   # meters (trigger for early slowdown and gentle turn)

            current_min_scan = np.min(scan)
            
            # Determine turn direction based on closest obstacle
            closest_ray_idx = np.argmin(scan)
            turn_direction = 1.0 if closest_ray_idx < LIDAR_RAYS / 2 else -1.0 # Turn away from obstacle

            if current_min_scan < collision_threshold:
                # Imminent collision: decisive but smooth turn with slight reverse
                print(f"⚠️ COLLISION IMMINENT! Min scan distance: {current_min_scan:.2f}. Initiating decisive evasive maneuver.")
                collision_avoidance_mode = True
                # Prioritize turning away, with a slight reverse to create space
                forward_speed = -1.0 # Very slight reverse
                turn_speed = turn_direction * 4.0 # More pronounced but still smooth turn
                apply_robot_action(robot_id, 0, forward_speed=forward_speed, turn_speed=turn_speed)
                # No explicit p.stepSimulation() loop here, let the main loop continue applying this action
            elif current_min_scan < warning_threshold:
                # Warning zone: gradual slowdown and gentle turn
                print(f"🚧 OBSTACLE AHEAD! Min scan distance: {current_min_scan:.2f}. Slowing down and adjusting course.")
                collision_avoidance_mode = True # Still in avoidance mode, but gentler
                # Proportional slowdown: closer to obstacle, slower speed
                # Linearly interpolate speed from 5.0 (at warning_threshold) to 0.0 (at collision_threshold)
                forward_speed = 5.0 * ((current_min_scan - collision_threshold) / (warning_threshold - collision_threshold))
                forward_speed = max(0.0, forward_speed) # Ensure non-negative speed
                turn_speed = turn_direction * 3.0 # Gentle turn, slightly more than before
                apply_robot_action(robot_id, 0, forward_speed=forward_speed, turn_speed=turn_speed)
            elif collision_avoidance_mode: # If previously in avoidance, but now clear
                collision_avoidance_mode = False
                exploration_mode = True # Transition to exploration
                exploration_step_count = 0
                print("✅ Collision avoided. Entering exploration mode.")
            # else: normal navigation logic (A* or RL) will apply

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
            
            # (Optional) Refine pose estimate with PL-ICP
            if prev_scan_points is not None:
                est_pose = pl_icp_correction(np.ascontiguousarray(scan_points), np.ascontiguousarray(prev_scan_points), est_pose)

            # Clip estimated pose to be within map boundaries
            est_pose[0] = np.clip(est_pose[0], 0, MAP_SIZE[0]-1)
            est_pose[1] = np.clip(est_pose[1], 0, MAP_SIZE[1]-1)
            
            # 2. MAPPING (SLAM)
            slam.update(est_pose, scan)

            # 3. VISUALIZE PARTICLES
            particles = mcl.get_particles()
            if particles:
                # Scale particle positions from map cells to meters for visualization
                # Assuming 15x15 meter environment and 300x300 map cells
                # Scaling factor = 15.0 / MAP_SIZE[0]
                scaling_factor = 15.0 / MAP_SIZE[0]

                positions = [[p.x * scaling_factor, p.y * scaling_factor, 0.1] for p in particles]
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
                        print(f"DEBUG: Estimated Pose: {est_pose}")
                        
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
                    print(f"DEBUG: Dist to target: {dist_to_target:.2f}")

                    delta_y = target_waypoint[1] - est_pose[1]
                    delta_x = target_waypoint[0] - est_pose[0]
                    desired_angle = np.arctan2(delta_y, delta_x)
                    angle_diff = (desired_angle - est_pose[2] + np.pi) % (2 * np.pi) - np.pi
                    print(f"DEBUG: Angle diff: {np.degrees(angle_diff):.2f} degrees")

                    # P-controller for turning
                    kp = 2.0
                    turn_speed = kp * angle_diff
                    print(f"DEBUG: Turn speed: {turn_speed:.2f}")

                    # Constant forward speed
                    forward_speed = 5.0

                    # Apply both forward and turn speeds
                    apply_robot_action(robot_id, 0, forward_speed=forward_speed, turn_speed=turn_speed)

                    if dist_to_target < 0.5: # If very close to waypoint, pop it
                        path.pop(0)
                        if not path:
                            action = 3 # Stop, path complete
                            time_taken = time.time() - start_time
                            print("\n" + "="*50)
                            print(f"🏁 GOAL REACHED! 🏁")
                            print(f"   - Time Taken: {time_taken:.2f} seconds")
                            print(f"   - Distance Traveled: {total_distance:.2f} meters")
                            print("="*50 + "\n")
                            apply_robot_action(robot_id, 3) # Stop the robot
                        else:
                            target_waypoint = path[0] # Update target

                else:
                    # No path found or goal reached, enter/continue exploration mode
                    if not exploration_mode:
                        exploration_mode = True
                        print("DEBUG: No path found to goal. Entering exploration mode to find unexplored regions.")
                    
                    # Find an unexplored target
                    current_grid_pos = (int(est_pose[0]), int(est_pose[1]))
                    unexplored_target = find_unexplored_target(grid_map, current_grid_pos)

                    if unexplored_target:
                        print(f"DEBUG: Found unexplored target: {unexplored_target}")
                        try:
                            # Plan path to unexplored target
                            path = astar_search(grid_map, current_grid_pos, unexplored_target)
                            if len(path) > 1:
                                print(f"📍 Path to unexplored found: {path[:3]}... (Total length: {len(path)})")
                                # Reset exploration mode if a path is found
                                exploration_mode = False
                                exploration_step_count = 0 # Reset exploration step count
                            else:
                                print(f"⚠️ No path found to unexplored target. Path: {path}. Trying simple exploration.")
                                # Fallback to simple exploration if path planning fails
                                # This part can be refined further for more sophisticated fallback
                                if exploration_step_count < 10: # Move forward for 10 steps
                                    action = 0 # Move forward
                                elif exploration_step_count < 15: # Then turn left for 5 steps
                                    action = 1 # Turn left
                                else:
                                    exploration_step_count = 0 # Reset cycle
                                    action = 3 # Stop for a moment
                                exploration_step_count += 1
                                apply_robot_action(robot_id, action)
                        except Exception as e:
                            print(f"⚠️ Path planning to unexplored failed: {e}. Trying simple exploration.")
                            # Fallback to simple exploration if path planning fails
                            if exploration_step_count < 10: # Move forward for 10 steps
                                action = 0 # Move forward
                            elif exploration_step_count < 15: # Then turn left for 5 steps
                                action = 1 # Turn left
                            else:
                                exploration_step_count = 0 # Reset cycle
                                action = 3 # Stop for a moment
                            exploration_step_count += 1
                            apply_robot_action(robot_id, action)
                    else:
                        print("DEBUG: No unexplored regions found. Performing simple exploration.")
                        # Fallback to simple exploration if no unexplored target is found
                        if exploration_step_count < 10: # Move forward for 10 steps
                            action = 0 # Move forward
                        elif exploration_step_count < 15: # Then turn left for 5 steps
                            action = 1 # Turn left
                        else:
                            exploration_step_count = 0 # Reset cycle
                            action = 3 # Stop for a moment
                        exploration_step_count += 1
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