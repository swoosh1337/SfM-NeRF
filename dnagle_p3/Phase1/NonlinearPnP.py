import numpy as np
from scipy.spatial.transform import Rotation 
import scipy.optimize as optimize
from utils import *


def non_linear_pnp(camera_matrix, image_points, object_points, initial_rotation, initial_translation):
    """
    Function to perform non-linear Perspective-n-Point (PnP) algorithm.

    :param camera_matrix: Camera matrix
    :param image_points: 2D points in image coordinates
    :param object_points: 3D points in object coordinates
    :param initial_rotation: Initial rotation matrix
    :param initial_translation: Initial translation vector

    :return: optimized_rotation: Optimized rotation matrix
             optimized_translation: Optimized translation vector
    """

    # Convert the initial rotation matrix to a quaternion
    initial_quaternion = get_quaternion(initial_rotation)

    # Construct the initial vector of parameters by concatenating the quaternion and translation vector
    initial_params = [initial_quaternion[0], initial_quaternion[1], initial_quaternion[2], initial_quaternion[3], 
                      initial_translation[0], initial_translation[1], initial_translation[2]]

    # Use the least_squares optimization method to find the optimized vector of parameters that minimizes the error function
    optimized_params = optimize.least_squares(
        fun=pnp_error_3,
        x0=initial_params,
        method="dogbox",
        args=[object_points, image_points, camera_matrix],
        verbose=0)

    optimized_vector = optimized_params.x

    # Extract the optimized quaternion and translation vector from the optimized vector of parameters
    optimized_quaternion = optimized_vector[:4]
    optimized_translation = optimized_vector[4:]

    # Convert the optimized quaternion to a rotation matrix
    optimized_rotation = get_rotation(optimized_quaternion)

    # Return the optimized rotation matrix and translation vector
    return optimized_rotation, optimized_translation



