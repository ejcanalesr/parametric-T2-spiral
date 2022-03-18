import nibabel as nib
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, inset_axes, mark_inset
from mpl_toolkits.axes_grid1 import make_axes_locatable

import warnings
warnings.filterwarnings("ignore",category=FutureWarning)

def colorbar(mappable):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax)
# end function

def plot_real_data_slices(path_to_save_data, path_to_data, Slice):
    print('Plotting quantitative maps')

    data_type = 'invivo'
    #_______________________________________________________________________________
    params = {
    'image.origin': 'lower',
    'image.interpolation': 'nearest',
    'image.cmap': 'gray',
    'axes.grid': False,
    'savefig.dpi': 600,  # to adjust notebook inline plot size
    'axes.labelsize': 12, # fontsize for x and y labels (was 10)
    'axes.titlesize': 14,
    'font.size': 14, # was 10
    'legend.fontsize': 12, # was 10
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'font.family': 'serif',
    }

    #'figure.figsize': [3.39, 2.10],

    #matplotlib.rcParams.update(params)

    fig1 = plt.figure('Showing all results', figsize=(11,10), constrained_layout=True)

    # load data
    img  = nib.load(path_to_data)
    data = img.get_data()
    data = data.astype(np.float64, copy=False)

    img  = nib.load(path_to_save_data + 'MWF.nii.gz')
    fM = img.get_data()
    fM = fM.astype(np.float64, copy=False)

    img  = nib.load(path_to_save_data + 'IEWF.nii.gz')
    fIE = img.get_data()
    fIE = fIE.astype(np.float64, copy=False)

    img  = nib.load(path_to_save_data + 'FWF.nii.gz')
    fCSF = img.get_data()
    fCSF = fCSF.astype(np.float64, copy=False)

    img  = nib.load(path_to_save_data + 'T2_M.nii.gz')
    T2m = img.get_data()
    T2m = T2m.astype(np.float64, copy=False)

    img  = nib.load(path_to_save_data + 'T2_IE.nii.gz')
    T2IE = img.get_data()
    T2IE = T2IE.astype(np.float64, copy=False)

    img  = nib.load(path_to_save_data + 'K_global_corrected.nii.gz')
    Ktotal = img.get_data()
    Ktotal = Ktotal.astype(np.float64, copy=False)

    #plt.subplot(2, 3, 1).set_axis_off()
    #im0 = plt.imshow(data[:,:,Slice,0].T, cmap='gray', origin='upper')
    #plt.title('Signal(TE=0)')
    #colorbar(im0)

    plt.subplot(2, 3, 1).set_axis_off()
    im6 = plt.imshow(Ktotal[:,:,Slice].T, cmap='gray', origin='upper')
    plt.title('Total Water Content')
    colorbar(im6)

    plt.subplot(2, 3, 2).set_axis_off()
    im1 = plt.imshow(fM[:,:,Slice].T, cmap='afmhot', origin='upper', clim=(0,0.25))
    plt.title('Myelin Water Fraction')
    colorbar(im1)

    plt.subplot(2, 3, 3).set_axis_off()
    im2 = plt.imshow(fIE[:,:,Slice].T, cmap='magma', origin='upper', clim=(0,1))
    plt.title('Intra/Extra Water Fraction')
    colorbar(im2)

    plt.subplot(2, 3, 4).set_axis_off()
    im3 = plt.imshow(fCSF[:,:,Slice].T, cmap='hot', origin='upper', clim=(0,1))
    plt.title('Free Water Fraction')
    colorbar(im3)

    plt.subplot(2, 3, 5).set_axis_off()
    im4 = plt.imshow(T2m[:,:,Slice].T, cmap='gnuplot2', origin='upper', clim=(5,20))
    plt.title('T2-Myelin (ms)')
    colorbar(im4)

    plt.subplot(2, 3, 6).set_axis_off()
    im5 = plt.imshow(T2IE[:,:,Slice].T, cmap='gnuplot2', origin='upper', clim=(30,90))
    plt.title('T2-Intra/Extra (ms)')
    colorbar(im5)

    #plt.tight_layout()
    fig1.set_constrained_layout_pads(w_pad=0.05, h_pad=0.05, hspace=0.07, wspace=-0.3)
    plt.savefig(path_to_save_data + 'Figure_slices.png', dpi=600)
    #plt.show()
    #fig1.show()

    #_______________________________________________________________________________
    fig2, axes = plt.subplots(nrows=2, ncols=3, figsize=(18,8.0), constrained_layout=True)
    ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()

    im1 = ax1.imshow(fM[:,:,Slice].T, cmap='gray', origin='upper', clim=(0,0.25))
    ax1.set_title('MWF')
    colorbar(im1)

    x = fM[:,:,Slice].flatten()
    x = x[x>0]
    ax4.hist(x, 50, density=1, facecolor='SkyBlue', alpha=1.0, range=[0, 0.4])
    ax4.set_title('Histogram of MWF')
    ax4.set_xlabel('MWF')
    ax4.set_ylabel('Probability')
    ax4.grid(True)


    im1 = ax2.imshow(T2m[:,:,Slice].T, cmap='gray', origin='upper', clim=(5,20))
    ax2.set_title('T2m')
    colorbar(im1)

    x = T2m[:,:,Slice].flatten()
    x = x[x>0]
    ax5.hist(x, 50, density=1, facecolor='IndianRed', alpha=1.0, range=[5, 20])
    ax5.set_title('Histogram of T2m')
    ax5.set_xlabel('T2m')
    ax5.set_ylabel('Probability')
    ax5.grid(True)

    im1 = ax3.imshow(T2IE[:,:,Slice].T, cmap='gray', origin='upper', clim=(30,90))
    ax3.set_title('T2IE')
    colorbar(im1)

    x = T2IE[:,:,Slice].flatten()
    x = x[x>0]
    ax6.hist(x, 50, density=1, facecolor='tan', alpha=1.0, range=[30, 110])
    ax6.set_title('Histogram of T2IE')
    ax6.set_xlabel('T2IE')
    ax6.set_ylabel('Probability')
    ax6.grid(True)

    #plt.tight_layout()
    plt.savefig(path_to_save_data + 'Figure_histograms.png', dpi=600)
    #plt.show()
    #fig2.show()
    #plt.close('all')
#end main function
