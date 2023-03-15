import numpy as np
from utils import *
import cv2
from EstimateFundamentalMatrix import *
from GetInliersRANSAC import *
from EssentialMatrixFromFundamentalMatrix import *


def extract_camera_pose(E):
    """
    Function to extract the camera pose from the essential matrix
    :param E: essential matrix
    :return: rotation matrix and camera center
    """
    # Perform singular value decomposition on essential matrix
    U, D, VT = np.linalg.svd(E) 
    # Define W matrix for camera pose estimation
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    # Compute four possible rotation matrices
    R = [U.dot(W.dot(VT)),
         U.dot(W.dot(VT)),
         U.dot(np.transpose(W).dot(VT)),
         U.dot(np.transpose(W).dot(VT))] 
    # Compute four possible camera centers
    C = [U[:, 2], -U[:, 2], U[:, 2], -U[:, 2]]
    # Enforce the determinant of R to be positive
    R = [-R[i] if (np.linalg.det(R[i]) < 0) else R[i] for i in range(4)]
    # Enforce the sign of the translation vector to match that of the camera center
    C = [-C[i] if (np.linalg.det(R[i]) < 0) else C[i] for i in range(4)]  
    return R, C



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
 
    filtered_feature_flag = np.zeros_like(feature_flag)

    f_matrix = np.empty(shape=(5,5), dtype=object)

    for i in range(0,1):
        for j in range(i+1,2):
            idx = np.where(feature_flag[:,i] & feature_flag[:,j])
            pts1, pts2 = read_matches_file(f"Data/P3Data/matching{i+1}.txt", j+1)     
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
    print("R_set: ", R_set)
    print("C_set: ", C_set)
    
     