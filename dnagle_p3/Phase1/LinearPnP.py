import numpy as np
from utils import *

def pnp(X_3d, x_2d, camera_matrix):
    """
    Function to compute camera pose from 3D-2D point correspondences
    :param X_3d: 3D points
    :param x_2d: 2D points
    :param camera_matrix: Camera matrix
    :return: R: Rotation matrix, C: Translation vector
    """
    N = X_3d.shape[0]
    
    # Homogenize 3D and 2D points
    X_4d = homogenize(X_3d)
    x_3d = homogenize(x_2d)
    
    # Normalize x by applying inverse of camera matrix K
    camera_matrix_inv = np.linalg.inv(camera_matrix)
    x_norm = camera_matrix_inv.dot(x_3d.T).T
    
    # Compute the A matrix
    for i in range(N):
        X = X_4d[i].reshape((1, 4))
        zeros = np.zeros((1, 4))
        
        u, v, _ = x_norm[i]
        
        # Compute the skew-symmetric matrix
        u_cross = np.array([[0, -1, v],
                            [1,  0 , -u],
                            [-v, u, 0]])
        
        # Compute the X_tilde matrix
        X_tilde = np.vstack((np.hstack((   X, zeros, zeros)), 
                            np.hstack((zeros,     X, zeros)), 
                            np.hstack((zeros, zeros,     X))))
        a = u_cross.dot(X_tilde)
        
        # Stack A vertically
        if i > 0:
            A = np.vstack((A, a))
        else:
            A = a
            
    # Compute the P matrix
    _, _, VT = np.linalg.svd(A)
    P = VT[-1].reshape((3, 4))
    
    # Compute the rotation matrix R
    R = P[:, :3]
    U_r, D, V_rT = np.linalg.svd(R) # Enforce orthonormality of R
    R = U_r.dot(V_rT)
    
    # Compute the translation vector C
    C = P[:, 3]
    C = - np.linalg.inv(R).dot(C)
    
    # Check for reflection case
    if np.linalg.det(R) < 0:
        R = -R
        C = -C
        
    return R, C




