import numpy as np
from utils import *
import cv2
from EstimateFundamentalMatrix import *
from GetInliersRANSAC import *
from EssentialMatrixFromFundamentalMatrix import *
from ExtractCameraPose import *
from DisambiguateCameraPose import *

def linear_triangulation(K, C1, R1, C2, R2, x1, x2):
    """
    Function to perform linear triangulation
    :param K: Camera matrix
    :param C1: Camera center 1
    :param R1: Rotation matrix 1
    :param C2: Camera center 2
    :param R2: Rotation matrix 2
    :param x1: points in image 1
    :param x2: points in image 2
    :return: 3D points
    """
    # Convert camera centers to 3x1 matrices
    C1 = np.reshape(C1, (3,1))
    C2 = np.reshape(C2, (3,1))

    # Compute projection matrices for each camera
    I = np.identity(3)
    P1 = np.dot(K, np.dot(R1, np.hstack((I,-C1))))
    P2 = np.dot(K, np.dot(R2, np.hstack((I,-C2))))

    # Extract individual rows of projection matrices
    p1T = P1[0,:].reshape(1,4)
    p2T = P1[1,:].reshape(1,4)
    p3T = P1[2,:].reshape(1,4)
    p_1T = P2[0,:].reshape(1,4)
    p_2T = P2[1,:].reshape(1,4)
    p_3T = P2[2,:].reshape(1,4)

    # Initialize list to store 3D points
    X = []

    # Loop over each point correspondence
    for i in range(x1.shape[0]):
        # Extract x, y coordinates of points in image 1 and image 2
        x = x1[i, 0]
        y = x1[i, 1]
        x_ = x2[i, 0]
        y_ = x2[i, 1]

        # Construct coefficient matrix
        A = []
        A.append((y * p3T) - p2T)
        A.append(p1T - (x * p3T))
        A.append((y_ * p_3T) - p_2T)
        A.append(p_1T - (x_ * p_3T))
        A = np.array(A).reshape(4,4)

        # Solve linear system using SVD
        _,_,vt = np.linalg.svd(A)
        v = vt.T
        x = v[:,-1]

        # Append 3D point to list
        X.append(x)

    # Convert list of 3D points to numpy array
    X = np.array(X)

    return np.array(X)





if __name__ == '__main__':
    img1 = cv2.imread('Data/P3Data/1.png')
    img2 = cv2.imread('Data/P3Data/2.png')


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

    print(pts1.shape)
    print(pts2.shape)
     
    #Compute Essential Matrix, Estimate Pose, Triangulate
    F = f_matrix[0,1]

    
    #K is given
    K = np.loadtxt('Data/P3Data/calibration.txt')
    E = essential_matrix_from_fundamental_matrix(K,F)

    #Estimating the Camera Pose
    R_set, C_set = extract_camera_pose(E)
    R1_ = np.identity(3)
    C1_ = np.zeros((3,1))

    pts3D_4 = []
    print(R_set[0].shape)
    print(C_set[0].shape)


    for i in range(len(C_set)):
        x1 = pts1
        x2 = pts2
        X = linear_triangulation(K, C1_, R1_, C_set[i], R_set[i], x1, x2)
        # X = linear_triangulation(R1_, R_set[i], C1_, C_set[i], K,  x1, x2)
        #Now we get 4 poses, we need to select unique one with maximum positive depth points
        X = X/X[:,3].reshape(-1,1)
        pts3D_4.append(X)


    plot_triangulation_3d(pts3D_4)
    R_best, C_best, X = disambiguate_pose(R_set,C_set,pts3D_4)
    plot_triangulation_3d([X])
    
    plot_reprojected_points(img1, pts1, pts2, pts3D_4[0], R_set[0], C_set[0], K)
