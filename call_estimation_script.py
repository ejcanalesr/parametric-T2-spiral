#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse

import numpy as np
from tabulate import tabulate

from recon_toolbox.estimate_multiexp_parallel_new import motor_recon
from recon_toolbox.plot_results_real_data import plot_real_data_slices

import time
import os
import warnings
warnings.filterwarnings("ignore",category=FutureWarning)

# ======================= Define input parameters ==============================
parser = argparse.ArgumentParser(description='Myelin Water Imaging')

parser.add_argument("--path_to_folder", default=None, type=str, help="Path to the folder where the data is located, e.g., /home/Datasets/T2spiral/", required=True)

parser.add_argument("--input", default=None, type=str, help="Input data, e.g., Data.nii.gz", required=True)

parser.add_argument("--mask", default=None, type=str, help="Brain mask, e.g., Mask.nii.gz", required=True)

parser.add_argument("--denoise",
                    choices=["TV2D", "TV3D", "None"],
                    required=True, type=str, default="None", help="Denoising method")

parser.add_argument("--reg_param", default=3.0, type=float, help="Regularization parameter for TV denoising")

parser.add_argument("--savefig",
                    choices=["yes", "no"],
                    required=True, type=str, default="yes", help="Save reconstructed maps in .png")

parser.add_argument("--savefig_slice", default=17, type=int, help="Axial slice to save reconstructed maps, e.g., --Slice=17", required=True)

parser.add_argument("--numcores", default=-1, type=int, help="Number of cores used in the computation: -1 = all cores")

args = parser.parse_args()

# ==============================================================================
path_to_folder   = args.path_to_folder
input_data       = args.input
mask             = args.mask
denoise          = args.denoise
reg_param        = args.reg_param
savefig          = args.savefig
Slice            = args.savefig_slice
num_cores        = args.numcores

# ==============================================================================
start_time = time.time()

path_to_data      = path_to_folder + input_data
path_to_mask      = path_to_folder + mask
path_to_save_data = path_to_folder

if denoise=="None":
    reg_param = 0
#end

TE_array = np.array([0, 7.6, 17.6, 67.6, 147.6, 307.6])
nTE      = len(TE_array)

headers =  [ 'Selected options             ',  '   '                   ]
table   =  [
           [ '1. Denoising method          ',  denoise,                ],
           [ '2. Regularization parameter  ',  reg_param,              ],
           [ '3. Number of TEs             ',  nTE,                    ],
           [ '4. TEs =                     ',  TE_array,               ],
           ]

#print('TEs =', TE_array)
#print(' ')

table_tabulated  = tabulate(table, headers=headers)
print ('-------------------------------')
print (table_tabulated)

try:
    os.mkdir(path_to_save_data)
except:
    print ('Warning: this folder already exists. Results will be overwritten')
#end try

motor_recon(TE_array, path_to_data, path_to_mask, path_to_save_data, denoise, reg_param, num_cores, nTE)

# ----------- PLOT reconstructed maps and spectra for a given Slice ------------
if savefig == 'yes':
    plot_real_data_slices(path_to_save_data, path_to_data, Slice)
# end

print("--- %s seconds ---" % (time.time() - start_time))
