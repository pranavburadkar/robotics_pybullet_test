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
    
    step_count = 0
    prev_pose = None
    prev_scan = None
    prev_scan_points = None
    path = []
    action = 3 # Default action is STOP

    try:
        while True:
            # Get current robot state
            current_pose = get_odometry(robot_id)
            scan, scan_points = simulate_lidar(robot_id, LIDAR_RAYS, LIDAR_RANGE, show_lasers=True)
            
            # Calculate motion since last step (odometry)
            if prev_pose is not None:
                delta_pose = current_pose - prev_pose
                delta_pose[2] = (delta_pose[2] + np.pi) % (2 * np.pi) - np.pi
            else:
                delta_pose = np.zeros(3)
            
            grid_map = slam.get_map() 
            
            # 1. LOCALIZATION (MCL)
            mcl.motion_update(delta_pose)
            mcl.measurement_update(scan, grid_map)
            est_pose = mcl.get_estimated_pose()
            
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
                        path = astar_search(grid_map, current_grid_pos, goal_coords)
                        if len(path) > 1:
                            print(f"📍 Path found: {path[:3]}...")
                        else:
                            path = []
                    except Exception as e:
                        print(f"⚠️ Path planning failed: {e}")
                        path = []

                if path:
                    target_waypoint = path[0]
                    dist_to_target = np.linalg.norm(np.array([est_pose[0], est_pose[1]]) - np.array(target_waypoint))

                    if dist_to_target < 1.0: # If close to waypoint
                        path.pop(0) # Move to next waypoint
                        if not path:
                            action = 3 # Stop, path complete
                            print("✅ Goal Reached!")
                        else:
                            target_waypoint = path[0] # Update target

                    if path:
                        delta_y = target_waypoint[1] - est_pose[1]
                        delta_x = target_waypoint[0] - est_pose[0]
                        desired_angle = np.arctan2(delta_y, delta_x)
                        angle_diff = (desired_angle - est_pose[2] + np.pi) % (2 * np.pi) - np.pi

                        if abs(angle_diff) > 0.2:
                            action = 1 if angle_diff > 0 else 2
                        else:
                            action = 0
                else:
                    action = 3 # No path

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