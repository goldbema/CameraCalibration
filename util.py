import numpy as np

def distort(k, normalized_proj):
    """Apply distortion to points in the normalized projection frame.

    Args:
       k: A vector of distortion coefficients [k0, k1]
       normalized_proj: An Nx2 ndarray of normalized projection points
    Returns:
       normalized projection points that have been radially distorted
    """

    x, y = normalized_proj[:, 0], normalized_proj[:, 1]

    # Calculate radii
    r = np.sqrt(x**2 + y**2)

    k0, k1 = k

    # Calculate distortion effects
    D = k0 * r**2 + k1 * r**4
    
    # Calculate distorted normalized projection values
    x_prime = x * (1. + D)
    y_prime = y * (1. + D)

    distorted_proj = np.hstack((x_prime[:, np.newaxis], y_prime[:, np.newaxis]))

    return distorted_proj

def reorthogonalize(R):
    """Determine least distance (Frobenius norm)rotation matrix 
       from a rotation matrix that has drifted away from orthogonality.

    Args:
       R: The matrix to reorthogonalize.
    Returns:
       The reorthogonalized matrix.
    """
    U, S, V_t = np.linalg.svd(R)
    new_R = np.dot(U, V_t)
    return new_R

def svd_solve(A):
    """Solve a homogeneous least squares problem with the SVD
       method.

    Args:
       A: Matrix of constraints.
    Returns:
       The solution to the system.
    """
    U, S, V_t = np.linalg.svd(A)
    idx = np.argmin(S)

    least_squares_solution = V_t[idx]

    return least_squares_solution

def to_homogeneous(A):
    """Convert a stack of inhomogeneous vectors to a homogeneous 
       representation.
    """
    A = np.atleast_2d(A)

    N = A.shape[0]
    A_hom = np.hstack((A, np.ones((N,1))))

    return A_hom

def to_inhomogeneous(A):
    """Convert a stack of homogeneous vectors to an inhomogeneous
       representation.
    """
    A = np.atleast_2d(A)

    N = A.shape[0]
    A /= A[:,-1][:, np.newaxis]
    A_inhom = A[:,:-1]

    return A_inhom

def to_homogeneous_3d(A):
    """Convert a stack of inhomogeneous vectors (without a Z component)
       to a homogeneous full-form representation.
    """
    if A.ndim != 2 or A.shape[-1] != 2:
        raise ValueError('Stacked vectors must be 2D inhomogeneous')

    N = A.shape[0]
    A_3d = np.hstack((A, np.zeros((N,1))))
    A_3d_hom = to_homogeneous(A_3d)

    return A_3d_hom

def project(K, k, E, model):
    """Project a point set into the sensor frame.

    Args:
       K: A 3x3 intrinsics matrix
       k: A vector of distortion parameters [k0, k1]
       E: A 3x4 extrinsics matrix
       model: An Nx2 point set (X,Y world coordinates)
    Returns:
       An Nx2 point set in the sensor frame.
    """

    model_hom = to_homogeneous_3d(model)

    normalized_proj = np.dot(model_hom, E.T)
    normalized_proj = to_inhomogeneous(normalized_proj)

    distorted_proj = distort(k, normalized_proj)
    distorted_proj_hom = to_homogeneous(distorted_proj)

    sensor_proj = np.dot(distorted_proj_hom, K[:-1].T)

    return sensor_proj