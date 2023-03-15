import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation 

def read_matches_file(filename, img_idx):
    """
    Function to read matches from file and extract the corresponding points
    :param filename: filename of the matches file
    :param img_idx: the index of the image we want to find matches for (e.g 2 means compare 1  with image 2 so it returns the points in image 1 that match with image 2 and vice versa)
    :return: two arrays of corresponding points
    """
    
    # Initialize arrays to store corresponding points
    pts1 = []
    pts2 = []
    
    # Initialize a set to keep track of duplicate points
    seen_pts = set()

    # Open matches file and read lines
    with open(filename, 'r') as f:
        lines = f.readlines()[1:] # skipping the first line
        
        # Loop through each line in the matches file
        for line in lines:
            
            # Split line into entries and extract number of matches
            entries = line.strip().split()
            n_matches = int(entries[0]) - 1
            
            # Loop through each match
            for i in range(n_matches):
                x1, y1 = float(entries[4]), float(entries[5])
                
                # Check if second image index matches the current image index
                if int(entries[i*3+6]) == img_idx: # accessing x2 and y2 based on i
                    x2, y2 = float(entries[i*3+7]), float(entries[i*3+8])
                    
                    # Check if corresponding point pair has already been seen
                    if (x1, y1, x2, y2) not in seen_pts:
                        pts1.append([x1, y1])
                        pts2.append([x2, y2])
                        seen_pts.add((x1, y1, x2, y2))

    # Convert point arrays to numpy arrays and return them as a tuple
    return np.array(pts1), np.array(pts2)



def features_extraction(data_folder_path):
    """
    Extracts features from the images in the specified folder.
    
    :param data_folder_path: Path to the folder containing image data.
    :return: feature_x, feature_y, feature_flag: Arrays containing extracted feature information.
    """

    # Initialize variables
    num_images = 5
    feature_x = []
    feature_y = []
    feature_flag = []

    # Loop through all matching files
    for n in range(1, num_images):
        file_path = data_folder_path + "/matching" + str(n) + ".txt"
        matching_file = open(file_path, "r")

        # Loop through each row in the matching file
        for i, row in enumerate(matching_file):
            if i == 0:
                # First row containing number of features
                row_elements = row.split(':')
            else:
                # Initialize arrays to store x, y, and flag values for each image
                x_row = np.zeros((1,num_images))
                y_row = np.zeros((1,num_images))
                flag_row = np.zeros((1,num_images), dtype=int)

                # Parse row elements and extract feature values
                row_elements = row.split()
                columns = [float(x) for x in row_elements]
                columns = np.asarray(columns)
                num_matches = columns[0]
             
                # Store x, y, and flag values for the current image
                current_x = columns[4]
                current_y = columns[5]
                x_row[0,n-1] = current_x
                y_row[0,n-1] = current_y
                flag_row[0,n-1] = 1

                # Loop through all matches for the feature
                m = 1
                while num_matches > 1:
                    image_id = int(columns[5+m])
                    image_id_x = int(columns[6+m])
                    image_id_y = int(columns[7+m])
                    m = m + 3
                    num_matches = num_matches - 1

                    # Store x, y, and flag values for the other images
                    x_row[0, image_id - 1] = image_id_x
                    y_row[0, image_id - 1] = image_id_y
                    flag_row[0, image_id - 1] = 1

                # Append x, y, and flag values for the feature to the corresponding lists
                feature_x.append(x_row)
                feature_y.append(y_row)
                feature_flag.append(flag_row)

    # Convert x, y, and flag values to numpy arrays and reshape to 2D arrays
    feature_x = np.asarray(feature_x).reshape(-1,num_images)
    feature_y = np.asarray(feature_y).reshape(-1,num_images)
    feature_flag = np.asarray(feature_flag).reshape(-1,num_images)

    # Return x, y, flag
    return feature_x, feature_y, feature_flag

def skew(x):
    '''
    Function to compute the skew matrix
    :param x: vector
    :return: skew matrix
    '''
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])

def estimate_epipole(F):
    """
    Function to estimate the epipole from the fundamental matrix
    :param F: Fundamental matrix
    :return: epipole
    """
    _,_,V = np.linalg.svd(F)
    e = V[-1,:]
    e /= e[-1]
    return e


def plot_epipolar_lines(F, x1, x2, img1, img2):
    """
    Function to plot epipolar lines
    :param F: Fundamental matrix
    :param x1: points in image 1
    :param x2: points in image 2
    :param img1: image 1
    :param img2: image 2
    :return: None
    """

    #get epipoles
    e1 = estimate_epipole(F)
    e2 = estimate_epipole(F.T)

    #create copies of the images
    img1_copy = img1.copy()
    img2_copy = img2.copy()

    #draw lines on the copies
    for pt in x1:
        img1_copy = cv2.line(img1_copy, tuple(pt.astype(int)), tuple(e1[:-1].astype(int)), (0,0,0), 1)
    for pt in x2:
        img2_copy = cv2.line(img2_copy, tuple(pt.astype(int)), tuple(e2[:-1].astype(int)), (0,0,0), 1)

    #display the copies
    cv2.imshow('1', img1_copy)
    cv2.waitKey(0)
    cv2.imshow('2', img2_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def projection_matrix(R,C,K):
    """
    Function to compute the projection matrix
    :param R: Rotation matrix
    :param C: Camera center
    :param K: Intrinsic matrix
    :return: Projection matrix
    """
    C = np.reshape(C,(3,1))
    I = np.identity(3)
    P = np.dot(K,np.dot(R,np.hstack((I,-C))))
    return P

def reprojection_error(X,pt1, pt2, R1, C1, R2, C2, K):
    """
    Compute the reprojection error between a 3D point and its corresponding 2D points in two images.
    
    :param X: 3D point in world coordinates
    :param pt1: 2D point in image 1
    :param pt2: 2D point in image 2
    :param R1: Rotation matrix of camera 1
    :param C1: Camera center of camera 1
    :param R2: Rotation matrix of camera 2
    :param C2: Camera center of camera 2
    :param K: Intrinsic matrix
    :return: Reprojection error in image 1 and image 2
    """
    
    # Compute projection matrices for camera 1 and camera 2
    P1 = projection_matrix(R1, C1, K)
    P2 = projection_matrix(R2, C2, K)
    
    # Extract rows of projection matrices and reshape them to 1x4 arrays
    P1_row1, P1_row2, P1_row3 = P1
    P1_row1, P1_row2, P1_row3 = P1_row1.reshape(1,4), P1_row2.reshape(1,4), P1_row3.reshape(1,4)
    P2_row1, P2_row2, P2_row3 = P2
    P2_row1, P2_row2, P2_row3 = P2_row1.reshape(1,4), P2_row2.reshape(1,4), P2_row3.reshape(1,4)

    # Convert 3D point to homogeneous coordinates
    X_homog = X.reshape(4,1)
    
    # Compute reprojection error in image 1
    u1, v1 = pt1[0], pt1[1]
    u1_proj = np.divide(P1_row1.dot(X_homog), P1_row3.dot(X_homog))
    v1_proj = np.divide(P1_row2.dot(X_homog), P1_row3.dot(X_homog))
    err1 = np.square(u1 - u1_proj) + np.square(v1 - v1_proj)

    # Compute reprojection error in image 2
    u2, v2 = pt2[0], pt2[1]
    u2_proj = np.divide(P2_row1.dot(X_homog), P2_row3.dot(X_homog))
    v2_proj = np.divide(P2_row2.dot(X_homog), P2_row3.dot(X_homog))
    err2 = np.square(u2 - u2_proj) + np.square(v2 - v2_proj)

    # Return both reprojection errors as output
    return err1, err2



def show_feature_matches(img1, img2, pts1, pts2):
    """
    Function to show the feature matches
    :param img1: image 1
    :param img2: image 2
    :param pts1: points in image 1
    :param pts2: points in image 2
    :return: None
    """
    #create image
    matches_img = np.hstack([img1, img2])
    
    #draw matches
    keypoints1 = [cv2.KeyPoint(pt[0], pt[1], 3) for pt in pts1]
    keypoints2 = [cv2.KeyPoint(pt[0], pt[1], 3) for pt in pts2]
    good_matches = [cv2.DMatch(index, index, 0) for index in range(len(pts1))]
    matches_img = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, matches_img, (0, 255, 0), (0, 0, 255))
		
    cv2.imshow("show_mathec",matches_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def projection_matrix(R, C, K):
    """
    Compute the projection matrix from the camera pose and the calibration matrix
    :param R: 3x3 rotation matrix
    :param C: 3x1 camera center
    :param K: 3x3 calibration matrix
    :return: 3x4 projection matrix
    """
    # Convert C to a column vector
    C = np.reshape(C, (-1, 1))

    # Compute the 3x4 projection matrix P = K [R | -C]
    Rt = np.hstack((R, -R.dot(C)))
    P = K.dot(Rt)

    return P


def show_inlier_matches(img1, img2, pts1, pts2, inliers):
    """
    Function to show the inlier matches
    :param img1: image 1
    :param img2: image 2
    :param pts1: points in image 1
    :param pts2: points in image 2
    :param inliers: indices of inlier matches
    :return: None
    """
    # Create image
    matches_img = np.hstack([img1, img2])

    # Draw inlier matches
    inlier_pts1 = pts1[inliers]
    inlier_pts2 = pts2[inliers]
    inlier_matches = [cv2.DMatch(idx, idx, 0) for idx in range(len(inliers))]
    matches_img = cv2.drawMatches(img1, [cv2.KeyPoint(x[0], x[1], 3) for x in inlier_pts1],
                                  img2, [cv2.KeyPoint(x[0], x[1], 3) for x in inlier_pts2],
                                  inlier_matches, matches_img, (0, 255, 0), (0, 0, 255))

    cv2.imshow("Ransac",matches_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def plot_reprojected_points(img1, pts1, pts2, pts3D, R, C, K, initial_color=(0, 250, 0), reproj_color=(0, 0, 255)):
    """
    Function to plot the reprojected points on the first image
    :param img1: image 1
    :param pts1: points in image 1
    :param pts2: points in image 2
    :param pts3D: 3D points
    :param R: 3x3 rotation matrix
    :param C: 3x1 camera center
    :param K: 3x3 calibration matrix
    :param initial_color: color of the points in the first image
    :param reproj_color: color of the reprojected points in the first image
    :return: None
    """
    # Reshape C to (3, 1)
    C = C.reshape(3, 1)

    # Compute the projection matrix
    Rt = np.hstack((R, -np.dot(R, C)))
    P = np.dot(K, Rt)

    # Extract (x, y, z) coordinates from pts3D
    pts3D_xyz = pts3D[:, :3]

    # Convert 3D points to homogeneous coordinates
    pts3D_homog = np.hstack((pts3D_xyz, np.ones((len(pts3D_xyz), 1))))

    # Project 3D points onto the first image
    proj_pts = np.dot(P, pts3D_homog.T)
    proj_pts = (proj_pts[:2, :] / proj_pts[2, :]).T

    # Draw circles around the corresponding points and reprojected points on the first image
    for pt1, proj_pt in zip(pts1, proj_pts):
        pt1 = tuple(map(int, pt1))
        proj_pt = tuple(map(int, proj_pt.reshape(-1)))
        cv2.circle(img1, pt1, 4, initial_color, -1)
        cv2.circle(img1, proj_pt, 4, reproj_color, -1)

    # Display the image with the corresponding points and reprojected points
    cv2.imshow('Reprojected points on first image', img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def plot_triangulation_3d(X_list):
    """
    Function to plot the 3D points in a 3D plot
    :param X_list: list of 3D points
    :return: None
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Extract x, y, and z coordinates from 3D points
    x_coords = np.concatenate([X[:, 0] for X in X_list])
    z_coords = np.concatenate([X[:, 2] for X in X_list])

    # Create a color array that maps each point to a color based on its camera pose
    color_array = np.zeros((x_coords.shape[0], 3))
    total_points = 0
    for i, X in enumerate(X_list):
        color = plt.cm.tab10(i)  # Get a unique color for each camera pose
        color_array[total_points:total_points+X.shape[0], :] = np.tile(color[:3], (X.shape[0], 1))
        total_points += X.shape[0]

    # Plot 3D points in 2D using different colors for different camera poses
    ax.scatter(x_coords, z_coords, s=1, c=color_array)

    # Set axis labels
    ax.set_xlabel('x')
    ax.set_ylabel('z')

    # Show the plot
    plt.show()

  
def homogenize(pts):
    """
    Function to convert 2D points to homogeneous coordinates
    :param pts: 2D points
    :return: 3D points in homogeneous coordinates
    """
    return np.hstack((pts, np.ones((pts.shape[0], 1))))



def pnp_error_3(X0, x3D, pts, K):
    """
    Function to compute reprojection error for PnP problem
    :param X0: 7x1 vector containing the quaternion and camera center
    :param x3D: 3D points
    :param pts: 2D points
    :param K: 3x3 calibration matrix
    :return: reprojection error
    """

    # Extract quaternion and camera center from X0
    Q, C = X0[:4], X0[4:].reshape(-1,1)

    # Compute rotation matrix from quaternion
    R = get_rotation(Q)

    # Compute projection matrix from rotation matrix, camera center, and calibration matrix
    P = projection_matrix(R,C,K)

    # Compute reprojection error for each 3D point and corresponding 2D point
    Error = []
    for X, pt in zip(x3D, pts):

        # Extract rows of projection matrix
        p_1T, p_2T, p_3T = P # rows of P

        # Reshape rows to 1x3 arrays
        p_1T, p_2T, p_3T = p_1T.reshape(1,-1), p_2T.reshape(1,-1), p_3T.reshape(1,-1)

        # Convert 3D point to homogeneous coordinates
        X = homogenize(X.reshape(1,-1)).reshape(-1,1)

        # Compute reprojection error for reference camera points
        u, v = pt[0], pt[1]
        u_proj = np.divide(p_1T.dot(X) , p_3T.dot(X))
        v_proj =  np.divide(p_2T.dot(X) , p_3T.dot(X))

        # Compute squared reprojection error
        E = np.square(v - v_proj) + np.square(u - u_proj)

        Error.append(E)

    # Compute mean reprojection error
    sumError = np.mean(np.array(Error).squeeze())
    return sumError



def pnp_error_3(camera_params, world_points, image_points, calibration_matrix):
    """
    Compute the reprojection error for a PnP problem.

    :param camera_params: 7x1 vector containing the quaternion and camera center
    :param world_points: 3D points in the world coordinate system
    :param image_points: corresponding 2D points in the image coordinate system
    :param calibration_matrix: 3x3 calibration matrix
    :return: mean squared reprojection error
    """

    # Extract quaternion and camera center from camera_params
    quaternion, camera_center = camera_params[:4], camera_params[4:].reshape(-1, 1)

    # Compute rotation matrix from quaternion
    rotation_matrix = get_rotation(quaternion)

    # Compute projection matrix from rotation matrix, camera center, and calibration matrix
    projection_matrix_ = projection_matrix(rotation_matrix, camera_center, calibration_matrix)

    # Compute reprojection error for each 3D point and corresponding 2D point
    errors = []
    for world_point, image_point in zip(world_points, image_points):

        # Extract rows of projection matrix
        row1, row2, row3 = projection_matrix_ # rows of P

        # Reshape rows to 1x3 arrays
        row1, row2, row3 = row1.reshape(1, -1), row2.reshape(1, -1), row3.reshape(1, -1)

        # Convert 3D point to homogeneous coordinates
        homogenous_point = homogenize(world_point.reshape(1, -1)).reshape(-1, 1)

        # Compute reprojection error for reference camera points
        u, v = image_point[0], image_point[1]
        u_proj = np.divide(row1.dot(homogenous_point), row3.dot(homogenous_point))
        v_proj = np.divide(row2.dot(homogenous_point), row3.dot(homogenous_point))

        # Compute squared reprojection error
        error = np.square(v - v_proj) + np.square(u - u_proj)

        errors.append(error)

    # Compute mean reprojection error
    mean_error = np.mean(np.array(errors).squeeze())
    return mean_error


def pnp_error(x3D_points, image_points, camera_matrix, rotation_matrix, camera_center):
    """
    Function to compute reprojection error for PnP
    :param x3D_points: 3D points
    :param image_points: 2D points
    :param camera_matrix: Camera matrix
    :param rotation_matrix: Rotation matrix
    :param camera_center: Camera center
    :return: reprojection error
    """
    # compute projection matrix
    projection_matrix_ = projection_matrix(rotation_matrix, camera_center, camera_matrix)
    
    # calculate reprojection error for each point
    errors = []
    for X, pt in zip(x3D_points, image_points):
        # convert X to a column of homogeneous vector
        X_homogeneous = homogenize(X.reshape(1, -1)).reshape(-1, 1)
        
        # calculate reprojection for the reference camera points
        p1, p2, p3 = projection_matrix_
        u_proj = np.divide(p1.dot(X_homogeneous), p3.dot(X_homogeneous))[0]
        v_proj = np.divide(p2.dot(X_homogeneous), p3.dot(X_homogeneous))[0]
        
        # calculate squared reprojection error
        u, v = pt[0], pt[1]
        squared_error = np.square(u - u_proj) + np.square(v - v_proj)
        errors.append(squared_error)

    # calculate mean error
    mean_error = np.mean(errors)
    
    return mean_error

def get_quaternion(R2):
    """
    Function to convert a rotation matrix to a quaternion
    :param R2: Rotation matrix
    return: Quaternion
    """

    # Convert rotation matrix to quaternion using the Rotation class from the scipy library
    Q = Rotation.from_matrix(R2)

    # Return quaternion as a numpy array
    return Q.as_quat()


def get_rotation(Q, type_ = 'q'):
    """
    Function to convert a quaternion to a rotation matrix
    :param Q: Quaternion
    :param type_: type of rotation matrix required, either 'q' or 'e'
    """

    if type_ == 'q':
        # Convert quaternion to rotation matrix using the Rotation class from the scipy library
        R = Rotation.from_quat(Q)
    elif type_ == 'e':
        # Convert euler angle to rotation matrix using the Rotation class from the scipy library
        R = Rotation.from_rotvec(Q)

    # Return rotation matrix as a numpy array
    return R.as_matrix()

def project(points_3d, camera_params, intrinsic_params):
    """
    Project 3D points onto the image plane
    :param points_3d: Numpy array containing 3D coordinates of points
    :param camera_params: Numpy array containing camera parameters
    :param intrinsic_params: Numpy array containing intrinsic camera parameters
    :return: Numpy array containing 2D coordinates of projected points
    """

    # Initialize list to hold projected points
    projected_points = []

    # Iterate over cameras
    for i in range(len(camera_params)):
        # Extract rotation and translation parameters
        rotation_matrix = get_rotation(camera_params[i, :3], 'e')
        translation_vector = camera_params[i, 3:].reshape(3, 1)

        # Compute projection matrix
        projection_matrix = np.dot(intrinsic_params, np.dot(rotation_matrix, np.hstack((np.identity(3), -translation_vector))))

        # Project points onto image plane
        homogeneous_3d_points = np.hstack((points_3d[i], 1)).T
        homogeneous_2d_points = np.dot(projection_matrix, homogeneous_3d_points)
        projected_2d_points = homogeneous_2d_points[:2] / homogeneous_2d_points[2]
        projected_points.append(projected_2d_points)

    return np.array(projected_points)


def get_euler(R2):
    """
    Function to convert a rotation matrix to a Euler angle vector
    :param R2: Rotation matrix
    return: Euler angle vector
    """
    euler = Rotation.from_matrix(R2)
    return euler.as_rotvec()


def pnp_error_2(feature_2d, point_3d, rotation_matrix, translation_vector, camera_matrix):
    """
    Function to compute the reprojection error for a given 3D point and its corresponding 2D feature
    :param feature_2d: 2D feature coordinates
    :param point_3d: 3D point coordinates
    :param rotation_matrix: Rotation matrix
    :param translation_vector: Translation vector
    :param camera_matrix: Camera matrix
    :return: Reprojection error
    """

    # Extract 2D feature coordinates
    u, v = feature_2d

    # Reshape 3D point to a 1x3 array and add homogeneous coordinate
    pts = point_3d.reshape(1,-1)
    point_3d_homogeneous = homogenize(pts).reshape(-1,1)

    # point_3d_homogeneous = np.hstack((pts, np.ones((pts.shape[0],1))))
    point_3d_homogeneous = point_3d_homogeneous.reshape(4,1)

    # Reshape camera center and compute projection matrix
    translation_vector = translation_vector.reshape(-1,1)
    projection_matrix_ = projection_matrix(rotation_matrix, translation_vector, camera_matrix)

    # Extract rows of projection matrix and reshape to 1x4 arrays
    row_1, row_2, row_3 = projection_matrix_
    row_1, row_2, row_3 = row_1.reshape(1,4), row_2.reshape(1,4), row_3.reshape(1,4)

    # Compute projected 2D feature coordinates
    u_proj = np.divide(row_1.dot(point_3d_homogeneous), row_3.dot(point_3d_homogeneous))
    v_proj = np.divide(row_2.dot(point_3d_homogeneous), row_3.dot(point_3d_homogeneous))
    projected_2d_feature = np.hstack((u_proj, v_proj))

    # Compute reprojection error as the norm of the difference between the observed and projected feature coordinates
    observed_2d_feature = np.hstack((u, v))
    reprojection_error = np.linalg.norm(observed_2d_feature - projected_2d_feature)

    return reprojection_error

def compute_residuals(x0, num_cameras, num_points, camera_indices, point_indices, observed_points, intrinsic_matrix):
    """
    Computes the residuals between observed and projected 2D points using camera parameters and 3D coordinates of points
    :param x0: Numpy array containing camera parameters and 3D coordinates of points
    :param num_cameras: Number of cameras in the scene
    :param num_points: Number of 3D points in the scene
    :param camera_indices: Numpy array containing camera indices for each observation
    :param point_indices: Numpy array containing point indices for each observation
    :param observed_points: Numpy array containing observed 2D points for each observation
    :param intrinsic_matrix: Numpy array containing intrinsic camera parameters
    :return: Numpy array containing residuals
    """
    num_total_cameras = num_cameras + 1  # Total number of cameras in the scene, including fixed camera
    camera_params = x0[:num_total_cameras * 6].reshape((num_total_cameras, 6))  # Extract camera parameters from x0 array
    points_3d = x0[num_total_cameras * 6:].reshape((num_points, 3))  # Extract 3D points from x0 array
    projected_points = project(points_3d[point_indices], camera_params[camera_indices], intrinsic_matrix)  # Project 3D points to 2D using camera params and intrinsic matrix
    residuals = (projected_points - observed_points).ravel()  # Compute residuals by subtracting projected 2D points from observed 2D points
    return residuals


def get_2d_points(X_index, visibility_matrix, feature_x, feature_y):
    """
    Get 2D points from the feature x and feature y having same index from the visibility matrix
    :param X_index: index of the 3D point in the feature matrix
    :param visibility_matrix: matrix indicating the visibility of the 3D point in different views
    :param feature_x: matrix containing x-coordinates of features in different views
    :param feature_y: matrix containing y-coordinates of features in different views
    :return: 2D points corresponding to the 3D point visible in all views
    """
    visible_feature_x = feature_x[X_index]
    visible_feature_y = feature_y[X_index]
    row_indices, col_indices = np.where(visibility_matrix == 1)
    pts_2d = np.column_stack((visible_feature_x[row_indices, col_indices], visible_feature_y[row_indices, col_indices]))
    return pts_2d


def get_camera_point_indices(visibility_matrix):
    """
    Function to extract camera and point indices from a visibility matrix
    :param visibility_matrix: a binary matrix indicating the visibility of points by cameras
    :return: camera_indices and point_indices, indicating the indices of cameras and points respectively
    """
    row_indices, col_indices = np.where(visibility_matrix == 1)
    camera_indices = col_indices
    point_indices = row_indices

    return camera_indices, point_indices

def rotate(points, rotation_vectors):
    """
    Rotates 3D points using Rodrigues' rotation formula
    :param points: Numpy array containing 3D coordinates of points
    :param rotation_vectors: Numpy array containing rotation vectors
    :return: Numpy array containing rotated 3D coordinates of points
    """
    theta = np.linalg.norm(rotation_vectors, axis=1)[:, np.newaxis]
    unit_rotation_vectors = np.divide(rotation_vectors, theta, out=np.zeros_like(rotation_vectors), where=theta != 0)
    dot_product = np.sum(points * unit_rotation_vectors, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rotated_points = (cos_theta * points) + (sin_theta * np.cross(unit_rotation_vectors, points)) + ((1 - cos_theta) * dot_product * unit_rotation_vectors)
    return rotated_points
