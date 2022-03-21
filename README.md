## Parametric T<sub>2</sub> relaxometry for myelin water quantification using T<sub>2</sub>-spiral data

<img src="Figure_slices.png" width="1082">

***The current implementation is written in Python 3***

## Install dependencies 🔧
```
- numpy
- nibabel
- matplotlib
- scipy
- skimage
- joblib
- multiprocessing
- progressbar
- joypy
- pandas
- tabulate
```

## Help 📖

Open a terminal and write:

```
$ python call_estimation_script.py -h

usage: call_estimation_script.py [-h] --path_to_folder PATH_TO_FOLDER --input INPUT --mask MASK --denoise {TV2D,TV3D,None} [--reg_param REG_PARAM] --savefig {yes,no} --savefig_slice SAVEFIG_SLICE
                                 [--numcores NUMCORES]

Myelin Water Imaging

optional arguments:
  -h, --help            show this help message and exit
  --path_to_folder PATH_TO_FOLDER
                        Path to the folder where the data is located, e.g., /home/Datasets/T2spiral/
  --input INPUT         Input data, e.g., Data.nii.gz
  --mask MASK           Brain mask, e.g., Mask.nii.gz
  --denoise {TV2D,TV3D,None}
                        Denoising method
  --reg_param REG_PARAM
                        Regularization parameter for TV denoising
  --savefig {yes,no}    Save reconstructed maps in .png
  --savefig_slice SAVEFIG_SLICE
                        Axial slice to save reconstructed maps, e.g., --Slice=90
  --numcores NUMCORES   Number of cores used in the computation: -1 = all cores

```

For more details see the example script: **script_run_T2spiral_fitting_example.sh** 🎁
We included some optional pre- and post-processing steps using FSL (https://fsl.fmrib.ox.ac.uk/fsl/fslwiki):

```
- Brain extraction for obtaining the brain mask (i.e., bet, FSL)
- Bias-field correction of the estimated proton density map, and segmentation to obtain WM, GM, and CSF probabilistic tissue-maps (i.e., fast, FSL)
```

## Estimated maps 🚀

The software will save the following metrics derived from the three-compartment model:

```
- MWF.nii.gz: Myelin Water Fraction
- IEWF.nii.gz: Intra- and Extra-cellular Water Fraction
- FWF.nii.gz: Free and quasi-free Water Fraction (i.e., T2 > 200ms) 
- T2_M.nii.gz: T2 of the myelin water.
- T2_IE.nii.gz: T2 of the intra- and extra-cellular water
- K_global.nii.gz: It is proportional to the Proton density (i.e., total water content)
- Predicted_Data4D.nii.gz: Predicted signal
```

## References ✒️
- **T<sub>2</sub>prep Three-Dimensional Spiral Imaging with Efficient Whole Brain Coverage for Myelin Water Quantification at 1.5 Tesla**
Thanh D. Nguyen, Cynthia Wisnieff, Mitchell A. Cooper, Dushyant Kumar, Ashish Raj, Pascal Spincemaille, Yi Wang, Tim Vartanian, and Susan A. Gauthier
Magnetic Resonance in Medicine 67:614–621 (2012)

- **Feasibility and Reproducibility of Whole Brain Myelin Water Mapping in 4 Minutes Using Fast Acquisition with Spiral Trajectory and Adiabatic T<sub>2</sub>prep (FAST-T<sub>2</sub>) at 3T**
Thanh D. Nguyen, Kofi Deh, Elizabeth Monohan, Sneha Pandya, Pascal Spincemaille, Ashish Raj, Yi Wang, and Susan A. Gauthier
Magn Reson Med. 2016 August ; 76(2): 456–465. doi:10.1002/mrm.25877.

- **Suppression of MRI Truncation Artifacts Using Total Variation Constrained Data Extrapolation**
Kai Tobias Block, Martin Uecker, and Jens Frahm
International Journal of Biomedical Imaging, Volume 2008, Article ID 184123, doi:10.1155/2008/184123

## Copyright and license 📄

**GNU Lesser General Public License v2.1**

Primarily used for software libraries, the GNU LGPL requires that derived works be licensed under the same license, but works that only link to it do not fall under this restriction.
