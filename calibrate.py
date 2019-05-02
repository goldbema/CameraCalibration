'''
An implementation of:

    Zhang, Z., "A Flexible New Technique for Camera Calibration," 
    IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 22,
    no. 11, Nov. 2000.

Example:

    Calculate intrinsic matrix, distortion coefficients, and extrinsic matrices
    for data points and planar model available in a directory:

        $ python calibrate.py --data_dir=[data dir]

'''

__author__ = 'Maxwell Goldberg'

import argparse
import numpy as np
import os

import dataloader
import distortion

import extrinsics
import homography
import intrinsics
import refinement
import util
import visualize

from scipy.optimize import curve_fit


def zhang_calibration(model, all_data):
    '''Perform camera calibration, including intrinsics, extrinsics,
       and distortion coefficients.

    Args:
       model: Nx2 collection of planar points in the world
       all_data: M-length list of Nx2 point sets of sensor correspondences
    Returns:
       Intrinsic matrix, distortion coefficients, and M-length list of 
       extrinsic matrices
    '''
    # model_hom = util.to_homogeneous(model)
    
    # Compute homographies for each image and run nonlinear refinement on each
    # homography
    homographies = []
    for data in all_data:
        H = homography.calculate_homography(model, data)
        H = homography.refine_homography(H, model, data)
        homographies.append(H)

    # Compute intrinsics
    K = intrinsics.recover_intrinsics(homographies)

    model_hom_3d = util.to_homogeneous_3d(model)

    # Compute extrinsics based on fixed intrinsics
    extrinsic_matrices = []
    for h, H in enumerate(homographies):
        E = extrinsics.recover_extrinsics(H, K)
        extrinsic_matrices.append(E)

        # Form projection matrix
        P = np.dot(K, E)

        predicted = np.dot(model_hom_3d, P.T)
        predicted = util.to_inhomogeneous(predicted)
        data = all_data[h]
        nonlinear_sse_decomp = np.sum((predicted - data)**2)

    # Calculate radial distortion based on fixed intrinsics and extrinsics
    k = distortion.calculate_lens_distortion(model, all_data, K, extrinsic_matrices)
    
    # Nonlinearly refine all parameters(intrinsics, extrinsics, and distortion)
    K_opt, k_opt, extrinsics_opt = refinement.refine_all_parameters(model, all_data, K, k, extrinsic_matrices)

    return K_opt, k_opt, extrinsics_opt


def parse_args():
    parser = argparse.ArgumentParser(description='Perform camera calibration')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the data directory')
    args = parser.parse_args()

    return args


def main(args):
    dataset = dataloader.load_dataset(args.data_dir)

    model, all_data = dataset['model'][0], dataset['data'] 

    K_opt, k_opt, extrinsics_opt = zhang_calibration(model, all_data)
    
    print('   Focal Length: [ {:.5f}  {:.5f} ]'.format(K_opt[0,0], K_opt[1,1]))
    print('Principal Point: [ {:.5f}  {:.5f} ]'.format(K_opt[0,2], K_opt[1,2]))
    print('           Skew: [ {:.5f} ]'.format(K_opt[0,1]))
    print('     Distortion: [ {:.5f}  {:.5f} ]'.format(k_opt[0], k_opt[1]))

    visualize.visualize_camera_frame(model, extrinsics_opt)
    visualize.visualize_world_frame(model, extrinsics_opt)

if __name__ == '__main__':
    args = parse_args()
    main(args)