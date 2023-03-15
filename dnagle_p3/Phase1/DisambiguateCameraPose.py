import numpy as np
from utils import *
import cv2
from EstimateFundamentalMatrix import *
from GetInliersRANSAC import *
from EssentialMatrixFromFundamentalMatrix import *
from ExtractCameraPose import *
from LinearTriangulation import *


def disambiguate_pose(rotation_matrices, camera_centers, three_d_points):
    """
    Function to select the best camera pose from the output of the PnP RANSAC algorithm.
    :param rotation_matrices: A list of rotation matrices
    :param camera_centers: A list of camera centers
    :param three_d_points: A list of 3D points
    :return: The best rotation matrix, camera center, and 3D points
    """
    # Initialize variables to store the best rotation matrix, camera center, and maximum positive depths
    best_index = 0
    max_positive_depths = 0

    # Loop over each possible camera pose
    for i, (rotation_matrix, camera_center, three_d_point) in enumerate(zip(rotation_matrices, camera_centers, three_d_points)):
        # Extract the third row of the rotation matrix
        r3 = rotation_matrix[2, :]

        # Normalize 3D points to obtain homogeneous coordinates
        three_d_point = three_d_point / three_d_point[:, 3][:, np.newaxis]
        three_d_point = three_d_point[:, :3]

        # Check if each 3D point is in front of the camera and has positive depth
        n_positive_depths = ((r3.dot((three_d_point - camera_center).T) > 0) & (three_d_point[:, 2] > 0)).sum()

        # Update best rotation matrix, camera center, and maximum positive depths
        if n_positive_depths > max_positive_depths:
            best_index = i
            max_positive_depths = n_positive_depths

    # Extract best rotation matrix, camera center, and 3D points
    best_rotation_matrix, best_camera_center, best_three_d_points = rotation_matrices[best_index], camera_centers[best_index], three_d_points[best_index]

    return best_rotation_matrix, best_camera_center, best_three_d_points


if __name__ == '__main__':

    pts1, pts2 = read_matches_file('Data/P3Data/matching1.txt',2)
    F = estimate_fundamental_matrix(pts1, pts2)
    images = []
    for i in range(1,3):
        path =  "Data/P3Data/" + str(i) + ".png"
        image = cv2.imread(path)
        if image is not None:
            images.append(image)
        else:
            print("No image is found")


    feature_x, feature_y, feature_flag =features_extraction("Data/P3Data")
 
    filtered_feature_flag = np.zeros_like(feature_flag) #np.zeros has limit which is solve by zeros_like
    # f_matrix = np.empty(shape=(5,5), dtype=object)
    f_matrix = np.empty(shape=(5,5), dtype=object)

    for i in range(0,1): #No of Images = 5
        for j in range(i+1,2):
            print("salam")

            idx = np.where(feature_flag[:,i] & feature_flag[:,j])
            pts1, pts2 = read_matches_file(f"Data/P3Data/matching{i+1}.txt", j+1)
            # pts1 = np.hstack((feature_x[idx,i].reshape((-1,1)), feature_y[idx,i].reshape((-1,1))))
            # pts2 = np.hstack((feature_x[idx,j].reshape((-1,1)), feature_y[idx,j].reshape((-1,1))))
            idx = np.array(idx).reshape(-1)
            
            if len(idx) > 8:
                F_inliers, inliers_idx = ransac(pts1,pts2,idx,2000,0.05)
               
                print("Between Images: ",i+1,"and",j+1,"NO of Inliers: ", len(inliers_idx), "/", len(idx) )
                f_matrix[i,j] = F_inliers
                filtered_feature_flag[inliers_idx,j] = 1
                filtered_feature_flag[inliers_idx,i] = 1


     
    F = f_matrix[0,1]
    K = np.loadtxt('Data/P3Data/calibration.txt')
    E = essential_matrix_from_fundamental_matrix(K,F)

    R_set, C_set = extract_camera_pose(E)
    R1_ = np.identity(3)
    C1_ = np.zeros((3,1))

    pts3D_4 = []
    for i in range(len(C_set)):
        x1 = pts1
        x2 = pts2
        "1, R2, C1, C2, K, pt1list, pt2list"
        X = linear_triangulation(K, C1_, R1_, C_set[i], R_set[i], x1, x2)
        X = X/X[:,3].reshape(-1,1)
        pts3D_4.append(X)


    R_best, C_best, X = disambiguate_pose(R_set,C_set,pts3D_4)
    print(pts3D_4)
    print([X])
    plot_triangulation_3d([X])
    