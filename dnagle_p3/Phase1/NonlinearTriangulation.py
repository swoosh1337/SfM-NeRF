import numpy as np
from utils import *
import cv2
from EstimateFundamentalMatrix import *
from GetInliersRANSAC import *
from EssentialMatrixFromFundamentalMatrix import *
from ExtractCameraPose import *
from DisambiguateCameraPose import *
from scipy import optimize



def nonLinear_triangulation(K, pts1, pts2, x3D, R1, C1, R2, C2):
    """
    Function to perform non-linear triangulation
    :param K: Camera matrix
    :param pts1: points in image 1
    :param pts2: points in image 2
    :param x3D: 3D points
    :param R1: Rotation matrix 1
    :param C1: Camera center 1
    :param R2: Rotation matrix 2
    :param C2: Camera center 2
    :return: 3D points
    """
    
    P1 = projection_matrix(R1,C1,K) # Compute projection matrix for camera 1
    P2 = projection_matrix(R2,C2,K) # Compute projection matrix for camera 2
    

    x3D_ = [] # Initialize list to store 3D points
    for i in range(len(x3D)):
        optimized_params = optimize.least_squares(fun=reprojeciton_loss, x0=x3D[i], method="trf", args=[pts1[i], pts2[i], P1, P2],verbose=False) # Non-linear optimization
        X1 = optimized_params.x # optimized 3D point
        x3D_.append(X1)  
    return np.array(x3D_)



def reprojeciton_loss(X, pts1, pts2, P1, P2):
    """
    Function to compute reprojection loss
    :param X: 3D point
    :param pts1: points in image 1
    :param pts2: points in image 2
    :param P1: Projection matrix 1
    :param P2: Projection matrix 2
    :return: reprojection loss
    """
    
    # Extract individual rows of projection matrices
    p1_1T, p1_2T, p1_3T = P1 # rows of P1
    p1_1T, p1_2T, p1_3T = p1_1T.reshape(1,-1), p1_2T.reshape(1,-1),p1_3T.reshape(1,-1) # reshape to 1x4

    p2_1T, p2_2T, p2_3T = P2 # rows of P2
    p2_1T, p2_2T, p2_3T = p2_1T.reshape(1,-1), p2_2T.reshape(1,-1), p2_3T.reshape(1,-1) 

    # Compute reprojection error for each point in image 1 and image 2
    u1,v1 = pts1[0], pts1[1] # u1,v1 are the coordinates of the point in image 1
    u1_proj = np.divide(p1_1T.dot(X) , p1_3T.dot(X)) # u1_proj is the projection of the point in image 1 
    v1_proj =  np.divide(p1_2T.dot(X) , p1_3T.dot(X)) # v1_proj is the projection of the point in image 1
    E1= np.square(v1 - v1_proj) + np.square(u1 - u1_proj)

    
    u2,v2 = pts2[0], pts2[1]
    u2_proj = np.divide(p2_1T.dot(X) , p2_3T.dot(X))
    v2_proj =  np.divide(p2_2T.dot(X) , p2_3T.dot(X))    
    E2= np.square(v2 - v2_proj) + np.square(u2 - u2_proj) # reprojection error for image 2
    
    error = E1 + E2 # total reprojection error
    return error.squeeze() # return error as a 1D array

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
    for i in range(len(C_set)):
        x1 = pts1
        x2 = pts2
        "1, R2, C1, C2, K, pt1list, pt2list"
        X = linear_triangulation(K, C1_, R1_, C_set[i], R_set[i], x1, x2)
        # X = linear_triangulation(R1_, R_set[i], C1_, C_set[i], K,  x1, x2)
        #Now we get 4 poses, we need to select unique one with maximum positive depth points
        X = X/X[:,3].reshape(-1,1)
        pts3D_4.append(X)


    R_best, C_best, X = disambiguate_pose(R_set,C_set,pts3D_4)
    X = X/X[:,3].reshape(-1,1)
    X_refined = nonLinear_triangulation(K,pts1,pts2,X,R1_,C1_,R_best,C_best)

    # plot_triangulation_3d([X_refined])
    plot_reprojected_points(img1, pts1, pts2, X_refined, R_set[0], C_set[0], K)