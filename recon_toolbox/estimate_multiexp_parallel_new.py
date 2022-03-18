#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import nibabel as nib
from numpy import inf

import scipy
from   scipy.optimize import minimize_scalar, fminbound, minimize

from   joblib import Parallel, delayed
import multiprocessing
import progressbar

from skimage.restoration import (denoise_tv_chambolle,
                                 estimate_sigma)
import os
import re

#import pdb

# -----------------------------------------------------------------------------#
# ---------------------------- Declare functions ------------------------------#
# -----------------------------------------------------------------------------#


# equality constraint (sum of volume fractions must be 1.0)
def con(x):
    return x[1] + x[3] + x[6] - 1
#end

def fitting_voxel(data_xyz, TE_array):
    #               x0  x1   x2  x3   x4   x5    x6
    x0          = [600, 0.1, 10, 0.89, 60, 2000, 0.01] # initial estimate for K1, MWF, T2_M, IEWF, T2_IE, T2_CSF, FWF
    #                   x0       x1      x2       x3        x4        x5          x6
    bnds        = ( (0, +inf), (0, 1), (10, 30), (0, 1), (31, 200), (201, 2000), (0, 1)) # bounds
    #res         = minimize(obj_fun, x0, method = 'L-BFGS-B', options={'gtol': 1e-20, 'disp': False, 'maxiter': 1000}, bounds = bnds, args=(data_xyz, TE_array) )
    res         = minimize(obj_fun, x0, method = 'SLSQP', options={'disp': False}, bounds = bnds, constraints=({'type':'eq', 'fun': con}), args=(data_xyz, TE_array) )
    reg_sol     = res.x
    return reg_sol[0], reg_sol[1], reg_sol[2], reg_sol[3], reg_sol[4], reg_sol[5], reg_sol[6]
#end fun

def obj_fun(x, data_xyz, TE_array):
    # Least square error
    term     = data_xyz - x[0] * ( x[1] * np.exp(-TE_array/x[2]) + x[3] * np.exp(-TE_array/x[4]) + x[6] * np.exp(-TE_array/x[5]) )
    cost_fun = np.sum(term**2)
    return cost_fun
#end fun

def fitting_slice(mask_1d, data_1d, nx, TE_array):
    # --------------------------------------
    nEchoes           = TE_array.shape[0]
    tmp_signal        = np.zeros((nx, nEchoes))
    tmp_K             = np.zeros((nx))
    tmp_MWF           = np.zeros((nx))
    tmp_T2_MWF        = np.zeros((nx))
    tmp_IEWF          = np.zeros((nx))
    tmp_T2_IE         = np.zeros((nx))
    tmp_FWF           = np.zeros((nx))
    tmp_T2_FWF        = np.zeros((nx))
    totVoxels_sclices = np.count_nonzero(mask_1d)
    if totVoxels_sclices > 0 :
        # ----------------------------------------------------------------------
        #                       Voxelwise estimation
        # ----------------------------------------------------------------------
        for voxelx in range(0, nx):
            if (mask_1d[voxelx] > 0.0) & (np.sum(data_1d[voxelx, :]) > 0.0):
                # ==================== Reconstruction
                data_xyz           = np.ascontiguousarray(data_1d[voxelx, :])
                K_xyz, MWF_xyz, T2_M_xyz, IEWF_xyz, T2_IE_xyz, T2_FWF_xyz, FWF_xyz = fitting_voxel(data_xyz, TE_array)
                predicted_data_xyz = K_xyz * ( MWF_xyz * np.exp(-TE_array/T2_M_xyz) + IEWF_xyz * np.exp(-TE_array/T2_IE_xyz) + FWF_xyz * np.exp(-TE_array/T2_FWF_xyz))
                # ---------------------------
                tmp_K[voxelx]        = K_xyz
                tmp_MWF[voxelx]      = MWF_xyz
                tmp_T2_MWF[voxelx]   = T2_M_xyz
                tmp_IEWF[voxelx]     = IEWF_xyz
                tmp_T2_IE[voxelx]    = T2_IE_xyz
                tmp_FWF[voxelx]      = FWF_xyz
                tmp_T2_FWF[voxelx]   = T2_FWF_xyz
                tmp_signal[voxelx,:] = predicted_data_xyz
            #end if mask
        #end for x
    #end if
    return tmp_K, tmp_MWF, tmp_T2_MWF, tmp_IEWF, tmp_T2_IE, tmp_FWF, tmp_T2_FWF, tmp_signal
#end main function

# Main function
def motor_recon(TE_array, path_to_data, path_to_mask, path_to_save_data, denoise, reg_param, num_cores, nTE):

    # Load Data and Mask
    img      = nib.load(path_to_data)
    data     = img.get_data()
    data     = data.astype(np.float64, copy=False)

    img_mask = nib.load(path_to_mask)
    mask     = img_mask.get_data()
    mask     = mask.astype(np.int64, copy=False)

    print('--------- Data shape -----------------')
    nx, ny, nz, nt = data.shape
    print(data.shape)
    if nTE != nt:
        print('*** Error: the number of volumes does not agree with the number of TEs ***')
        sys.exit()
    #end if
    print('--------------------------------------')

    for c in range(nt):
        data[:,:,:,c] = np.squeeze(data[:,:,:,c]) * mask
    #end

    # Only for testing: selects a few slices
    #mask[:,:,18:-1] = 0
    #mask[:,:,0:17]  = 0

    # -------------------------------------------------------------------------#
    # ---------------------------- Denoising ----------------------------------#
    # -------------------------------------------------------------------------#

    if denoise == 'TV3D' :
        print ('Step #1: Denoising using (3D) Total Variation:')
        for voxelt in progressbar.progressbar(range(nt), redirect_stdout=True):
            print(voxelt+1, ' volumes processed')
            data_vol  = np.squeeze(data[:,:,:,voxelt])
            sigma_est = np.mean(estimate_sigma(data_vol, multichannel=False))
            data[:,:,:,voxelt] = denoise_tv_chambolle(data_vol, weight=reg_param*sigma_est, eps=0.0002, n_iter_max=200, multichannel=False)
        #end for
        outImg = nib.Nifti1Image(data, img.affine)
        nib.save(outImg, path_to_save_data + 'Data_denoised.nii.gz')
    #end if
    if denoise == 'TV2D' : #Preferred option for this data
        print ('Step #1: Denoising using (2D) Total Variation:')
        # This could be a better option if the slice thickness is much bigger than the in plane resolution
        for voxelt in progressbar.progressbar(range(nt), redirect_stdout=True):
            print(voxelt+1, ' volumes processed')
            for slicet in range(nz):
                data_vol2D  = np.squeeze(data[:,:,slicet,voxelt])
                if np.sum(np.abs(data_vol2D).flatten()) > 0 :
                    sigma_est   = estimate_sigma(data_vol2D)
                    data[:,:,slicet,voxelt] = denoise_tv_chambolle(data_vol2D, weight=reg_param*sigma_est, eps=0.0002, n_iter_max=200)
                #end
            #end for slices
        # end for volumes
        outImg = nib.Nifti1Image(data, img.affine)
        nib.save(outImg, path_to_save_data + 'Data_denoised.nii.gz')
    #end if

    # -------------------------------------------------------------------------#
    # -------------------------- Estimation -----------------------------------#
    # -------------------------------------------------------------------------#
    print ('Step #2: Voxelwise Estimation:')

    K_global = np.zeros((nx, ny, nz))
    MWF      = np.zeros((nx, ny, nz))
    T2_M     = np.zeros((nx, ny, nz))
    IEWF     = np.zeros((nx, ny, nz))
    T2_IE    = np.zeros((nx, ny, nz))
    FWF      = np.zeros((nx, ny, nz))
    T2_FWF   = np.zeros((nx, ny, nz))
    Predicted_Data4D = np.zeros((nx, ny, nz, nt))

    number_of_cores = multiprocessing.cpu_count()
    if num_cores == -1:
        num_cores = number_of_cores
        print ('Using all CPUs: ', number_of_cores)
    else:
        print ('Using ', num_cores, ' CPUs from ', number_of_cores)
    #end if

    for voxelz in progressbar.progressbar(range(nz), redirect_stdout=True):
        print(voxelz+1, ' slices processed')
        # Parallelization by rows: this is more efficient for computing a single or a few slices
        mask_slice = mask[:,:,voxelz]
        data_slice = data[:,:,voxelz,:]
        Estimated_parameters = Parallel(n_jobs=num_cores, backend='multiprocessing')(delayed(fitting_slice)(mask_slice[:, voxely], data_slice[:,voxely,:], nx, TE_array) for voxely in range(ny))
        for voxely in range(ny):
            K_global[:,voxely,voxelz] = Estimated_parameters[voxely][0]
            MWF[:,voxely,voxelz]      = Estimated_parameters[voxely][1]
            T2_M[:,voxely,voxelz]     = Estimated_parameters[voxely][2]
            IEWF[:,voxely,voxelz]     = Estimated_parameters[voxely][3]
            T2_IE[:,voxely,voxelz]    = Estimated_parameters[voxely][4]
            FWF[:,voxely,voxelz]      = Estimated_parameters[voxely][5]
            T2_FWF[:,voxely,voxelz]   = Estimated_parameters[voxely][6]
            Predicted_Data4D[:,voxely,voxelz,:] = Estimated_parameters[voxely][7]
        #end voxely
    #end voxelx

    # -------------------------------------------------------------------------#
    # -------------------------- Save results ---------------------------------#
    # -------------------------------------------------------------------------#

    print ('Step #3: Save to disk:')
    outImg = nib.Nifti1Image(K_global, img.affine)
    print('Saving K_global map to :', path_to_save_data + 'K_global.nii.gz')
    nib.save(outImg, path_to_save_data + 'K_global.nii.gz')

    outImg = nib.Nifti1Image(MWF, img.affine)
    print('Saving MWF map to :', path_to_save_data + 'MWF.nii.gz')
    nib.save(outImg, path_to_save_data + 'MWF.nii.gz')

    outImg = nib.Nifti1Image(T2_M, img.affine)
    print('Saving T2_M map to :', path_to_save_data + 'T2_M.nii.gz')
    nib.save(outImg, path_to_save_data + 'T2_M.nii.gz')

    outImg = nib.Nifti1Image(IEWF, img.affine)
    print('Saving IEWF map to :', path_to_save_data + 'IEWF.nii.gz')
    nib.save(outImg, path_to_save_data + 'IEWF.nii.gz')

    outImg = nib.Nifti1Image(T2_IE, img.affine)
    print('Saving T2_IE map to :', path_to_save_data + 'T2_IE.nii.gz')
    nib.save(outImg, path_to_save_data + 'T2_IE.nii.gz')

    outImg = nib.Nifti1Image(T2_FWF, img.affine)
    print('Saving T2_FWF map to :', path_to_save_data + 'T2_FWF.nii.gz')
    nib.save(outImg, path_to_save_data + 'T2_FWF.nii.gz')

    outImg = nib.Nifti1Image(FWF, img.affine)
    print('Saving FWF map to :', path_to_save_data + 'FWF.nii.gz')
    nib.save(outImg, path_to_save_data + 'FWF.nii.gz')

    outImg = nib.Nifti1Image(Predicted_Data4D, img.affine)
    print('Saving predicted signal to :', path_to_save_data + 'Predicted_Data4D.nii.gz')
    nib.save(outImg, path_to_save_data + 'Predicted_Data4D.nii.gz')

    print ('Done!')
#end main function
