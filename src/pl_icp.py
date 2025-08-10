import numpy as np
from numba import jit

@jit(nopython=True)
def pl_icp_correction(current_scan_points, previous_scan_points, pose_guess, iterations=10):
    """Basic 2D Point-to-Point ICP for pose correction."""
    # Extract x, y coordinates, ignoring z for 2D ICP
    current_points = current_scan_points[:, :2]
    previous_points = previous_scan_points[:, :2]

    # Initial transformation from pose_guess
    # Convert pose_guess (x, y, yaw) to a 2D transformation matrix
    # R = [[cos(yaw), -sin(yaw)], [sin(yaw), cos(yaw)]]
    # T = [x, y]
    current_x, current_y, current_yaw = pose_guess
    
    # Initialize transformation (R, t) for ICP based on pose_guess
    R = np.array([[np.cos(current_yaw), -np.sin(current_yaw)],
                  [np.sin(current_yaw),  np.cos(current_yaw)]])
    t = np.array([current_x, current_y])

    # Apply initial transformation to current_points
    transformed_current_points = (R @ current_points.T).T + t

    for i in range(iterations):
        # Find closest points (correspondences)
        # For each point in transformed_current_points, find its nearest neighbor in previous_points
        correspondences = []
        for p_curr in transformed_current_points:
            diff = previous_points - p_curr
            distances = np.sqrt(np.sum(np.square(diff), axis=1))
            closest_idx = np.argmin(distances)
            
            # Add a correspondence threshold
            if distances[closest_idx] < 0.5: # Threshold for good correspondence
                correspondences.append((p_curr, previous_points[closest_idx]))
        
        # Only perform update if enough correspondences are found
        if len(correspondences) < 5: # Minimum number of correspondences
            break # No correspondences found or too few good ones

        # Separate corresponding points
        # Pre-allocate arrays for Numba compatibility
        P = np.empty((len(correspondences), 2), dtype=np.float64)
        Q = np.empty((len(correspondences), 2), dtype=np.float64)

        for k in range(len(correspondences)):
            P[k] = correspondences[k][0]
            Q[k] = correspondences[k][1]

        # Compute centroids
        centroid_P = np.array([np.mean(P[:, 0]), np.mean(P[:, 1])])
        centroid_Q = np.array([np.mean(Q[:, 0]), np.mean(Q[:, 1])])

        # Center the points
        P_centered = P - centroid_P
        Q_centered = Q - centroid_Q

        # Compute covariance matrix H
        H = P_centered.T @ Q_centered

        # SVD of H
        U, S, Vt = np.linalg.svd(H)

        # Compute rotation matrix
        R_new = Vt.T @ U.T

        # Special reflection case
        if np.linalg.det(R_new) < 0:
            Vt[-1, :] *= -1
            R_new = Vt.T @ U.T

        # Compute translation vector
        t_new = centroid_Q - R_new @ centroid_P

        # Update overall transformation
        R = R_new @ R
        t = R_new @ t + t_new

    # Convert final R, t back to (dx, dy, dtheta) relative to initial pose_guess
    # This is a simplified conversion for 2D
    final_yaw_correction = np.arctan2(R[1, 0], R[0, 0])
    final_dx = t[0]
    final_dy = t[1]

    # Apply correction to the initial pose_guess
    # This assumes the ICP finds the transformation from current_points to previous_points
    # and we want to apply this correction to the robot's pose.
    # The pose_guess is (x, y, yaw) of the robot.
    # The ICP finds the transformation from the current scan (in robot frame) to the previous scan (in world frame).
    # So, we need to apply this transformation to the robot's pose.
    
    # For a simple 2D case, we can directly add the corrections.
    # A more rigorous approach would involve composing transformations.
    corrected_x = pose_guess[0] + final_dx
    corrected_y = pose_guess[1] + final_dy
    corrected_yaw = pose_guess[2] + final_yaw_correction

    return np.array([corrected_x, corrected_y, corrected_yaw])
