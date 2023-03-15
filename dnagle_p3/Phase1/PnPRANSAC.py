import numpy as np
from LinearPnP import *
from utils import *


def pnp_ransac(camera_matrix, features_2d, points_3d, num_iterations=2000, inlier_threshold=2):
    """
    PnP RANSAC algorithm to estimate camera pose from 2D-3D correspondences
    :param camera_matrix: camera matrix
    :param features_2d: 2D image points
    :param points_3d: 3D world points
    :param num_iterations: number of RANSAC iterations
    :param inlier_threshold: maximum distance in pixels for a correspondence to be considered an inlier
    :return: R_best, t_best
    """

    # Initialize variables for RANSAC
    best_inlier_count = 0
    R_best, t_best = None, None
    num_points = points_3d.shape[0]

    # Run RANSAC iterations
    for i in range(num_iterations):

        # Select random 6 3D-2D point correspondences
        rand_indices = np.random.choice(num_points, size=6)
        x_set, X_set = features_2d[rand_indices], points_3d[rand_indices]

        # Compute camera pose using PnP
        R, t = pnp(X_set, x_set, camera_matrix)

        # Check number of inliers using PnP error
        inlier_indices = []
        if R is not None:
            for j in range(num_points):
                feature = features_2d[j]
                X = points_3d[j]
                error = pnp_error_2(feature, X, R, t, camera_matrix)

                if error < inlier_threshold:
                    inlier_indices.append(j)

        # Update RANSAC if current iteration has more inliers
        if len(inlier_indices) > best_inlier_count:
            best_inlier_count = len(inlier_indices)
            R_best = R
            t_best = t

    return R_best, t_best

                