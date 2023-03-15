import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
from utils import *


DEFAULT_MATCHES_FOLDER = 'P3Data'
DEFAULT_CALIBRATION_FILENAME = 'P3Data/calibration.txt'



def estimate_fundamental_matrix(corresponding_points_image_1, corresponding_points_image_2):
    """
    Computes the fundamental matrix from corresponding points (x1,x2) in two images.

    :param corresponding_points_image_1: (N,2) array of corresponding (x,y) coordinates from the first image.
    :param corresponding_points_image_2: (N,2) array of corresponding (x,y) coordinates from the second image.
    :return: (3,3) fundamental matrix
    """

    # Extract x and y coordinates from corresponding points
    x1_coordinates, y1_coordinates = corresponding_points_image_1[:, 0], corresponding_points_image_1[:, 1]
    x2_coordinates, y2_coordinates = corresponding_points_image_2[:, 0], corresponding_points_image_2[:, 1]

    # Create an array of ones to represent the homogeneous coordinate
    homogeneous_coordinates = np.ones(x1_coordinates.shape[0])

    # Construct the A matrix
    A = [x1_coordinates * x2_coordinates, y1_coordinates * x2_coordinates, x2_coordinates, 
         x1_coordinates * y2_coordinates, y1_coordinates * y2_coordinates, y2_coordinates, 
         x1_coordinates, y1_coordinates, homogeneous_coordinates]  # N x 9
    A = np.vstack(A).T  # N x 9

    # Compute the SVD of A and extract the last row of V
    U, D, V = np.linalg.svd(A)
    fundamental_matrix_vector = V[-1, :]

    # Reshape the vectorized fundamental matrix to a 3x3 matrix
    fundamental_matrix_noisy = fundamental_matrix_vector.reshape(3, 3)

    # Enforce the rank-2 constraint on the fundamental matrix using SVD cleanup
    UF, UD, UV = np.linalg.svd(fundamental_matrix_noisy)
    UD[-1] = 0
    fundamental_matrix = UF @ np.diag(UD) @ UV
    fundamental_matrix /= fundamental_matrix[2, 2]

    return fundamental_matrix



if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--matches_folder_path', default=DEFAULT_MATCHES_FOLDER)
    argparser.add_argument('--calib_file_path', default=DEFAULT_CALIBRATION_FILENAME)
    args = argparser.parse_args()

    img1 = cv2.imread('Data/P3Data/1.png')
    img2 = cv2.imread('Data/P3Data/2.png')

    pts1, pts2 = read_matches_file('Data/P3Data/matching1.txt',2)
    print(len(pts1),"matches found")
    print(pts2,"<-- pts2")
    F = estimate_fundamental_matrix(pts1, pts2)
    print(F,"<-- F\n")
    plot_epipolar_lines(F,pts1,pts2,img1,img2)
    show_feature_matches(img1,img2,pts1,pts2)
