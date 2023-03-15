import numpy as np
import time
import cv2
from scipy.spatial.transform import Rotation
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares
from BuildVisibilityMatrix import *
from utils import *



def bundle_adjustment_sparsity(X_found, filtered_feature_flag, num_cameras):
    
    """
    Functuin to compute the sparsity pattern of the Jacobian matrix for bundle adjustment optimization.
    :param X_found: Numpy array containing 3D coordinates of points
    :param filtered_feature_flag: Numpy array containing flags indicating whether a point is a feature or not
    :param num_cameras: Number of cameras in the scene
    :return: Sparse matrix containing the sparsity pattern of the Jacobian matrix
    """
    
    num_camera_params = num_cameras + 1
    X_index, visibility_matrix = find_visible_cam(X_found.reshape(-1), filtered_feature_flag, num_cameras)
    num_observations = np.sum(visibility_matrix)
    num_points = len(X_index[0])

    num_rows = num_observations * 2
    num_cols = num_camera_params * 6 + num_points * 3   # We refine only orientation and translation of 3D points and not camera parameters, such as focal length and distortion.
    A = lil_matrix((num_rows, num_cols), dtype=int)
   
    obs_indices = np.arange(num_observations)
    camera_indices, point_indices = get_camera_point_indices(visibility_matrix)

    # Fill the rows of A corresponding to the camera parameters
    for i in range(6):
        A[2 * obs_indices, camera_indices * 6 + i] = 1
        A[2 * obs_indices + 1, camera_indices * 6 + i] = 1

    # Fill the rows of A corresponding to the 3D point parameters
    for i in range(3):
        A[2 * obs_indices, (num_cameras) * 6 + point_indices * 3 + i] = 1
        A[2 * obs_indices + 1, (num_cameras) * 6 + point_indices * 3 + i] = 1

    return A



def bundle_adjustment(X_index, visibility_matrix, X_all, X_found, feature_x, feature_y, filtered_feature_flag, R_set, C_set, K, n_cameras):
    """
    Function to perform bundle adjustment on the 3D points and camera parameters.
    :param X_index: Indices of 3D points that are visible in the current camera
    :param visibility_matrix: Visibility matrix for the current camera
    :param X_all: Numpy array containing 3D coordinates of points
    :param X_found: Numpy array containing boolean values that correspond to whether a 3D point is found or not
    :param feature_x: Numpy array containing x coordinates of 2D features
    :param feature_y: Numpy array containing y coordinates of 2D features
    :param filtered_feature_flag: Numpy array containing flags that correspond to the presence of 2D features in the image
    :param R_set: List of rotation matrices for each camera
    :param C_set: List of translation vectors for each camera
    :param K: Intrinsic camera matrix
    :param n_cameras: Number of cameras in the scene
    :return: R_set: List of optimized rotation matrices for each camera
    :return: C_set: List of optimized translation vectors for each camera
    :return: X_all: Numpy array containing optimized 3D coordinates of points
    """
    # Extract 3D points and 2D feature points from the dataset.
    points_3d = X_all[X_index]
    points_2d = get_2d_points(X_index, visibility_matrix, feature_x, feature_y)

    # Extract rotation and translation parameters for each camera.
    camera_params = []
    for i in range(n_cameras + 1):
        C, R = C_set[i], R_set[i]
        euler_angles = get_euler(R)
        camera_params_ = [euler_angles[0], euler_angles[1], euler_angles[2], C[0], C[1], C[2]]
        camera_params.append(camera_params_)
    # Convert list of camera parameters into 1D array.
    camera_params = np.array(camera_params, dtype=object).reshape(-1, 6)

    # Create initial guess for camera and 3D point parameters by stacking 1D arrays.
    x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))
    n_points = points_3d.shape[0]

    # Extract camera and point indices for each observation.
    camera_indices, point_indices = get_camera_point_indices(visibility_matrix)

    # Create sparsity matrix for bundle adjustment.
    sparsity_matrix = bundle_adjustment_sparsity(X_found, filtered_feature_flag, n_cameras)

    # Run bundle adjustment using least squares optimization.
    t0 = time.time()
    result = least_squares(
        compute_residuals,
        x0,
        jac_sparsity=sparsity_matrix,
        verbose=0,
        x_scale='jac',
        ftol=1e-10,
        method='trf',
        args=(n_cameras, n_points, camera_indices, point_indices, points_2d, K),
    )
    t1 = time.time()

    # Extract optimized camera and 3D point parameters from the optimization result.
    x1 = result.x
    n_cams = n_cameras + 1
    optim_cam_params = x1[:n_cams * 6].reshape((n_cams, 6))
    optim_points_3d = x1[n_cams * 6:].reshape((n_points, 3))

    # Create optimized 3D point array.
    optim_X_all = np.zeros_like(X_all)
    optim_X_all[X_index] = optim_points_3d

    # Convert optimized camera parameters into rotation and translation matrices.
    optim_C_set, optim_R_set = [], []
    for i in range(len(optim_cam_params)):
        R = get_rotation(optim_cam_params[i, :3], 'e')
        C = optim_cam_params[i, 3:].reshape(3, 1)
        optim_C_set.append(C)
        optim_R_set.append(R)

    return optim_R_set, optim_C_set, optim_X_all
