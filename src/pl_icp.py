import numpy as np

def pl_icp_correction(lidar_pts_t, lidar_pts_tp, pose_guess):
    # Simulate a PL-ICP pose adjustment (here, just do minimal correction)
    # In real ICP, we'd align point clouds and estimate (dx, dy, dtheta).
    # To reflect the paper, add a small random correction to your pose estimate.
    dx = np.random.normal(0, 0.02)
    dy = np.random.normal(0, 0.02)
    dtheta = np.random.normal(0, 0.01)
    new_pose = (
        pose_guess[0] + dx,
        pose_guess[1] + dy,
        pose_guess[2] + dtheta
    )
    return new_pose
