import cv2
import numpy as np

import util

from scipy.optimize import curve_fit

def pack_params(K, k, extrinsic_matrices):
    """Pack parameters into raveled representation.
    """
    packed_params = []

    # Flatten intrinsics
    alpha, beta, gamma, u_c, v_c = K[0,0], K[1,1], K[0,1], K[0,2], K[1,2]
    k0, k1 = k
    a = [alpha, beta, gamma, u_c, v_c, k0, k1]

    packed_params.extend(a)

    # Flattened extrinsics
    for E in extrinsic_matrices:
        # Convert extrinsics to flattened Rodrigues representation
        R = E[:3, :3]
        t = E[:, 3]

        rodrigues = cv2.Rodrigues(R)[0]

        rho_x, rho_y, rho_z = rodrigues
        t_x, t_y, t_z = t

        e = [rho_x, rho_y, rho_z, t_x, t_y, t_z]

        packed_params.extend(e)

    packed_params = np.array(packed_params)

    return packed_params

def unpack_refinement_params(params):
    """Unpack intrinsics, distortion parameters, and extrinsics
       from raveled representation.
    """
    intrinsics = params[:7]

    # Unpack intrinsics
    alpha, beta, gamma, u_c, v_c, k0, k1 = intrinsics
    K = np.array([[alpha, gamma, u_c],
                  [   0.,  beta, v_c],
                  [   0.,    0.,  1.]])
    k = np.array([k0, k1])

    # Unpack extrinsics
    extrinsic_matrices = []
    for i in range(7, len(params), 6):
        E_rodrigues = params[i:i+6]
        rho_x, rho_y, rho_z, t_x, t_y, t_z = E_rodrigues
        R = cv2.Rodrigues(np.array([rho_x, rho_y, rho_z]))[0]
        t = np.array([t_x, t_y, t_z])

        E = np.zeros((3, 4))
        E[:3, :3] = R
        E[:, 3] = t

        extrinsic_matrices.append(E)

    return K, k, extrinsic_matrices

def f_refine_all(xdata, *params):
    """Value function for Levenberg-Marquardt refinement.
    """
    if len(params) < 7 or len(params[7:]) % 6 != 0:
        raise ValueError('Check parameter vector encoding')
    if xdata.ndim != 1:
        raise ValueError('Check data vector encoding')

    intrinsics = params[:7]

    M = len(params[7:]) // 6
    N = xdata.shape[0] // (2 * M)

    # Unpack data vector
    X = xdata[:N]
    Y = xdata[N:2*N]
    model = np.zeros((N, 2))
    model[:, 0] = X
    model[:, 1] = Y
    
    # model_hom = util.to_homogeneous_3d(model)

    # Unpack parameters
    K, k, extrinsic_matrices = unpack_refinement_params(params)

    # Form observation vectors
    obs_x = np.zeros(N*M)
    obs_y = np.zeros(N*M)
    for e, E in enumerate(extrinsic_matrices):
        # Project the model into the sensor frame for each image, and append the 
        # predicted points to the observation vectors
        sensor_proj = util.project(K, k, E, model)

        x, y = sensor_proj[:, 0], sensor_proj[:, 1]

        obs_x[e*N:(e+1)*N] = x
        obs_y[e*N:(e+1)*N] = y

    # Stack observation vectors. Note that we observe the same convention as before
    # in stacking all x observations prior to all y observations
    result = np.zeros(2*N*M)
    result[:N*M] = obs_x
    result[N*M:] = obs_y

    return result

def refine_all_parameters(model, all_data, K, k, extrinsic_matrices):
    """Perform nonlinear refinement of all parameters using linear
       estimates as preconditioning

    Args:
       model: Nx2 points in the planar world frame model
       all_data: M-length list of Nx2 points in the sensor frame for
                 each of M images
       K: A 3x3 intrinsics matrix
       k: A 2-vector of distortion parameters
       extrinsic_matrices: M-length list of 3x4 extrinsics matrices
    Returns:
       Optimized values for K, k, and all extrinisc matrices
    """
    M = len(all_data)
    N = model.shape[0]
    
    p0 = pack_params(K, k, extrinsic_matrices)

    # Flatten model into predictor variable
    X, Y = model[:,0], model[:,1]
    xdata_ = np.hstack((X, Y))
    xdata = np.zeros(2*M*N)
    xdata[:N] = X
    xdata[N:2*N] = Y

    # Flatten data into dependent variable
    obs_x, obs_y = [], []
    for data in all_data:
        x, y = data[:,0], data[:,1]
        
        obs_x.append(x)
        obs_y.append(y)

    obs_x = np.hstack(obs_x)
    obs_y = np.hstack(obs_y)
    
    ydata = np.hstack((obs_x, obs_y))

    # Minimize reprojection error with Levenberg-Marquardt
    popt, pcov = curve_fit(f_refine_all, xdata, ydata, p0)

    # Unpack parameters
    K_opt, k_opt, extrinsic_matrices_opt = unpack_refinement_params(popt)

    return K_opt, k_opt, extrinsic_matrices_opt