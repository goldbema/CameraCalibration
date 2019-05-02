import numpy as np

import util

from scipy.optimize import curve_fit


def calculate_normalization_matrix(data):
    """Calculates zero-centered, sqrt(2) distance transformed
       matrix transform for data.

       Args:
          data: Nx2 stack of data points
        Returns:
          The normalization matrix
    """
    if data.ndim != 2 or data.shape[-1] != 2:
        raise ValueError('Dataset must be a collection of 2D points')

    x, y = data[:, 0], data[:, 1]

    N = data.shape[0]

    x_mean, y_mean = x.mean(), y.mean()
    x_var, y_var = x.var(), y.var()
    
    # Form rescaling matrix so that data points will lie
    # sqrt(2) from the origin on average.
    s_x, s_y = np.sqrt(2. / x_var), np.sqrt(2. / y_var)
    
    norm_matrix = np.array([[s_x,  0., -s_x * x_mean],
                            [ 0., s_y, -s_y * y_mean],
                            [ 0.,  0.,            1.]])

    return norm_matrix

def calculate_homography(model, data):
    """Perform linear least squares to calculate homography between planar 
       model and sensor data
    """
    N = model.shape[0]

    # Normalize data
    norm_matrix_model = calculate_normalization_matrix(model)
    norm_matrix_data  = calculate_normalization_matrix(data)

    model = util.to_homogeneous(model)
    data =  util.to_homogeneous(data)

    model_norm = np.dot(model, norm_matrix_model.T)
    data_norm = np.dot(data, norm_matrix_data.T)

    X, Y, x, y = model_norm[:,0], model_norm[:,1], data_norm[:,0], data_norm[:,1]
    
    # Mount homogeneous constraint matrix (See Burger pg. 11-13 for a derivation)
    A = np.zeros((N * 2, 9))

    x_component = np.zeros((N, 9))
    x_component[:, 0] = -X
    x_component[:, 1] = -Y
    x_component[:, 2] = -1.
    x_component[:, 6] =  x * X
    x_component[:, 7] =  x * Y
    x_component[:, 8] =  x

    y_component = np.zeros((N, 9))
    y_component[:, 3] = -X
    y_component[:, 4] = -Y
    y_component[:, 5] = -1.
    y_component[:, 6] =  y * X
    y_component[:, 7] =  y * Y
    y_component[:, 8] =  y

    # Note that all x-constraints precede all y-constraints for convenience of 
    # representation.
    A[:N] = x_component
    A[N:] = y_component

    # Solve homogeneous system
    h_norm = util.svd_solve(A)

    # Reconstitute normalized homography
    H_norm = h_norm.reshape((3,3))

    # Denormalize
    H = np.dot(np.dot(np.linalg.inv(norm_matrix_data), H_norm), norm_matrix_model)

    return H


def f_refine(xdata, *params):
    """Value function for Levenberg-Marquardt refinement.
    """
    h11, h12, h13, h21, h22, h23, h31, h32, h33 = params

    N = xdata.shape[0] // 2

    X = xdata[:N]
    Y = xdata[N:]

    x = (h11 * X + h12 * Y + h13) / (h31 * X + h32 * Y + h33)
    y = (h21 * X + h22 * Y + h23) / (h31 * X + h32 * Y + h33)

    result = np.zeros_like(xdata)
    result[:N] = x
    result[N:] = y

    return result


def jac_refine(xdata, *params):
    """Jacobian function for Levenberg-Marquardt refinement.
    """
    h11, h12, h13, h21, h22, h23, h31, h32, h33 = params

    N = xdata.shape[0] // 2

    X = xdata[:N]
    Y = xdata[N:]

    J = np.zeros((N * 2, 9))
    J_x = J[:N]
    J_y = J[N:]

    s_x = h11 * X + h12 * Y + h13
    s_y = h21 * X + h22 * Y + h23
    w   = h31 * X + h32 * Y + h33
    w_sq = w**2

    J_x[:, 0] = X / w
    J_x[:, 1] = Y / w
    J_x[:, 2] = 1. / w
    J_x[:, 6] = (-s_x * X) / w_sq
    J_x[:, 7] = (-s_x * Y) / w_sq
    J_x[:, 8] = -s_x / w_sq

    J_y[:, 3] = X / w
    J_y[:, 4] = Y / w
    J_y[:, 5] = 1. / w
    J_y[:, 6] = (-s_y * X) / w_sq
    J_y[:, 7] = (-s_y * Y) / w_sq
    J_y[:, 8] = -s_y / w_sq

    J[:N] = J_x
    J[N:] = J_y

    return J


def refine_homography(H, model, data):
    """Perform nonlinear least squares to refine linear homography
       estimates.

    Args:
       H: 3x3 homography matrix
       model: Nx2 world frame planar model
       data: Nx2 sensor frame correspondences
    Returns:
       Refined 3x3 homography
    """
    X, Y, x, y = model[:,0], model[:,1], data[:,0], data[:,1]

    N = X.shape[0]

    h0 = H.ravel()

    xdata = np.zeros(N * 2)
    xdata[:N] = X
    xdata[N:] = Y

    ydata = np.zeros(N * 2)
    ydata[:N] = x
    ydata[N:] = y

    # Use Levenberg-Marquardt to refine the linear homography estimate
    popt, pcov = curve_fit(f_refine, xdata, ydata, p0=h0, jac=jac_refine)
    h_refined = popt

    # Normalize and reconstitute homography
    h_refined /= h_refined[-1]
    H_refined = h_refined.reshape((3,3))

    return H_refined