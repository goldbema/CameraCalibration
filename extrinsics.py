import numpy as np

import util

def recover_extrinsics(H, K):
    """Use computed homography and intrinsic matrix to calculate
       corresponding extrinsic matrix

    Args:
       H: 3x3 homography matrix
       K: 3x3 intrinsic matrix
    Returns:
       3x4 extrinsic matrix
    """
	# Obtain column vectors from homography matrix
    h0, h1, h2 = H[:,0], H[:,1], H[:,2]

    K_inv = np.linalg.inv(K)

    # Form normalizer
    lambda_ = 1. / np.linalg.norm(np.dot(K_inv, h0))
    
    # Compute r0, r1, and t from the homography. r2 can be derived
    # by an orthogonality constraint
    r0 = lambda_ * np.dot(K_inv, h0)
    r1 = lambda_ * np.dot(K_inv, h1)
    r2 = np.cross(r0, r1)
    t  = lambda_ * np.dot(K_inv, h2)

    # Reconstitute the rotation component of the extrinsics and reorthogonalize
    R = np.vstack((r0, r1, r2)).T
    R = util.reorthogonalize(R)

    # Reconstitute full extrinsics
    E = np.hstack((R, t[:, np.newaxis]))

    return E