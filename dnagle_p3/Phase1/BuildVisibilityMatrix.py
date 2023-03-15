import numpy as np


def find_visible_cam(X_found, filtered_feature_flag, current_camera_index):
    """"
    Function to find the indices of 3D points that are visible in the current camera
    :param X_found: A numpy array containing boolean values that correspond to whether a 3D point is found or not
    :param filtered_feature_flag: A numpy array containing flags that correspond to the presence of 2D features in the image
    :param current_camera_index: The index of the current camera
    :return: visible_3d_point_indices, visibility_matrix
    """
    # find the 3d points such that they are visible in either of the cameras < current_camera_index
    # Create a temporary binary array to store the features that are visible in the current camera or any previous cameras
    temp_feature_flag = np.zeros((filtered_feature_flag.shape[0]), dtype = int)
    for n in range(current_camera_index + 1):
        temp_feature_flag = temp_feature_flag | filtered_feature_flag[:,n]

    # Get the indices of 3D points that are visible in the current camera
    visible_3d_point_indices = np.where((X_found.reshape(-1)) & (temp_feature_flag))

    # Get a visibility matrix to store the presence of 2D features for each 3D point
    visibility_matrix = X_found[visible_3d_point_indices].reshape(-1,1)
    for n in range(current_camera_index + 1):
        visibility_matrix = np.hstack((visibility_matrix, filtered_feature_flag[visible_3d_point_indices, n].reshape(-1,1)))

    _, c = visibility_matrix.shape
    return visible_3d_point_indices, visibility_matrix[:, 1:c]


