import time
import numpy as np
from config import MAP_SIZE, LIDAR_RAYS, LIDAR_RANGE, PARTICLE_COUNT, RL_ACTIONS
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
    
    # Goal coordinates for path planning - FIXED
    start_coords = (MAP_SIZE[0]//2, MAP_SIZE[1]//2)
    goal_coords = (13, 13)
    
    print("="*50)
    print("🤖 ASSIGNMENT 2: Multi-Sensor Robot Navigation")
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
    
    try:
        while True:
            # Get current robot state
            current_pose = get_odometry(robot_id)
            scan = simulate_lidar(robot_id, LIDAR_RAYS, LIDAR_RANGE, show_lasers=True)
            
            # Calculate motion since last step (odometry)
            if prev_pose is not None:
                delta_pose = current_pose - prev_pose
                # Handle angle wrap-around for yaw
                delta_pose[2] = (delta_pose[2] + np.pi) % (2 * np.pi) - np.pi
            else:
                delta_pose = np.zeros(3)
            
            # Get the map from the previous state for localization
            grid_map = slam.get_map() 
            
            # 1. LOCALIZATION (MCL)
            # Predict particle movement based on odometry
            mcl.motion_update(delta_pose)
            # Update particle weights based on LIDAR scan against the current map
            mcl.measurement_update(scan, grid_map)
            est_pose = mcl.get_estimated_pose()
            
            # (Optional) Refine pose estimate with PL-ICP
            if prev_scan is not None:
                est_pose = pl_icp_correction(scan, prev_scan, est_pose)
            
            # 2. MAPPING (SLAM)
            # Update the occupancy grid map using the new best pose estimate and scan
            slam.update(est_pose, scan)
            
            # 3. EXPLORATION (RL)
            # The RL agent decides the next action to maximize exploration.
            num_unknown = np.sum(grid_map == 0)  # Unexplored cells
            reward = num_unknown / (MAP_SIZE[0] * MAP_SIZE[1])  # Exploration reward
            action = rl_agent.select_action(0)  # Simple state
            apply_robot_action(robot_id, action)
            rl_agent.update(action, reward)
            
            # 4. PATH PLANNING (A*)
            # Periodically, plan a path from the current estimated position to the goal.
            # Note: The robot is currently driven by the RL exploration agent, not this path.
            if step_count % 30 == 0:
                try:
                    current_grid_pos = (int(est_pose[0]), int(est_pose[1]))
                    path = astar_search(grid_map, current_grid_pos, goal_coords)
                    if len(path) > 1:
                        print(f"📍 Path found: {path[:3]}... (showing first 3 waypoints)")
                    else:
                        print("🚫 No path to goal found")
                except Exception as e:
                    print(f"⚠️  Path planning failed: {e}")
            
            # Update state for next iteration
            prev_pose = current_pose.copy()
            prev_scan = scan.copy()
            
            # Print status every 10 steps
            if step_count % 10 == 0:
                print(f"Step {step_count:4d} | Pose: ({est_pose[0]:5.2f}, {est_pose[1]:5.2f}, {est_pose[2]:5.2f}) | "
                      f"RL Q-values: {np.round(rl_agent.q_table, 2)} | Action: {action}")
            
            # Step physics simulation and sleep
            step_count += 1
            p.stepSimulation()
            time.sleep(1/60)  # 60 Hz simulation
            
    except KeyboardInterrupt:
        print("\n🛑 Simulation stopped by user.")
        print("📊 Final Statistics:")
        print(f"   - Total steps: {step_count}")
        print(f"   - Final pose: ({est_pose[0]:.2f}, {est_pose[1]:.2f}, {est_pose[2]:.2f})")
        print(f"   - Map coverage: {np.sum(grid_map > 0.1)}/{MAP_SIZE[0]*MAP_SIZE[1]} cells")
        p.disconnect()

if __name__ == "__main__":
    run()
