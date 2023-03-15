import numpy as np
from utils import *
import cv2
from EstimateFundamentalMatrix import *
from GetInliersRANSAC import *


def essential_matrix_from_fundamental_matrix(K, F):
    """
    Function to calculate the essential matrix from the fundamental matrix and the calibration matrix
    :param K: calibration matrix
    :param F: fundamental matrix
    :return: essential matrix
    """
    # Calculate essential matrix from calibration matrix and fundamental matrix
    E = np.transpose(K).dot(F.dot(K))
    # Perform singular value decomposition on E
    u, s, vh = np.linalg.svd(E)
    # Ensure that E is of rank 2
    s = np.diag([1, 1, 0])
    # Reconstruct E from u, s, vh
    E = u.dot(s.dot(vh))
    return E



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

    for i in range(0,1): #No of Images = 5
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
    F12 = f_matrix[0,1]
    K = np.loadtxt('Data/P3Data/calibration.txt')
    E = essential_matrix_from_fundamental_matrix(K,F12)
    print(E)