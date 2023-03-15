import cv2
import numpy as np
from EstimateFundamentalMatrix import *

def ransac(pts1, pts2, idx, num_iterations, error_threshold):
    """
    RANSAC algorithm for estimating the fundamental matrix.

    :param pts1: array of shape (N, 2) containing 2D points in image 1
    :param pts2: array of shape (N, 2) containing 2D points in image 2
    :param idx: array of shape (N,) containing indices of corresponding points
    :param num_iterations: maximum number of iterations to run RANSAC
    :param error_threshold: maximum allowed error for a point to be considered an inlier
    :return: tuple containing the estimated fundamental matrix and indices of inlier points
    """
    
    inliers_threshold = 0
    inliers_indices = []
    f_inliers = None

    for i in range(num_iterations):
        # Select 8 random correspondences
        n_rows = pts1.shape[0]
        rand_indxs = np.random.choice(n_rows, 8)
        x1 = pts1[rand_indxs, :]
        x2 = pts2[rand_indxs, :]
        F = estimate_fundamental_matrix(x1, x2)
        indices = []

        if F is not None:
            for j in range(n_rows):
                x1_j = np.array([pts1[j, 0], pts1[j, 1], 1])
                x2_j = np.array([pts2[j, 0], pts2[j, 1], 1]).T
                error = np.abs(np.dot(x2_j, np.dot(F, x1_j)))
                if error < error_threshold:
                    indices.append(idx[j])

        if len(indices) > inliers_threshold:
            inliers_threshold = len(indices)
            inliers_indices = indices
            f_inliers = F  # Choose F with maximum number of inliers.

    return f_inliers, inliers_indices



if __name__ == '__main__':

    pts1, pts2 = read_matches_file('Data/P3Data/matching1.txt',2)
    F = estimate_fundamental_matrix(pts1, pts2)
    images = []
    for i in range(1,3): #5 images given
        path =  "Data/P3Data/" + str(i) + ".png"
        image = cv2.imread(path)
        if image is not None:
            images.append(image)
        else:
            print("No image is found")

    feature_x, feature_y, feature_flag  =features_extraction("Data/P3Data")
 
    filtered_feature_flag = np.zeros_like(feature_flag) #
    f_matrix = np.empty(shape=(5,5), dtype=object)

    for i in range(0,1): 
        for j in range(i+1,2):
            print("salam")

            idx = np.where(feature_flag[:,i] & feature_flag[:,j])
            pts1, pts2 = read_matches_file(f"Data/P3Data/matching{i+1}.txt", j+1)
            idx = np.array(idx).reshape(-1)
            
            if len(idx) > 8:
                F_inliers, inliers_idx = ransac(pts1,pts2,idx,2000,0.05)
               
                print("Between Images: ",i+1,"and",j+1,"NO of Inliers: ", len(inliers_idx), "/", len(idx) )
                f_matrix[i,j] = F_inliers
                filtered_feature_flag[inliers_idx,j] = 1
                filtered_feature_flag[inliers_idx,i] = 1
