"""
Created on Wed June 12 11:42:25 2024
Correlation matrix generator for spheres
"""

import os
import h5py
import numpy as np
import scipy
import matplotlib as mpl 
import matplotlib.pyplot as plt
from scipy import fft
from sklearn.decomposition import PCA ## need this for principle component analysis below
from matplotlib.mlab import psd
from scipy.optimize import minimize, curve_fit
from scipy.stats import chi2
from scipy.signal import welch
from scipy.signal.windows import blackman
from scipy.signal import find_peaks, butter, lfilter
from matplotlib.pyplot import gca
import pandas as pd
from matplotlib.colors import LogNorm
import seaborn as sn
from scipy.interpolate import interp1d

#from folderscanning_videoprocessor import *
#from correlationcalculator import *
from basichdf5 import *


def array_single_file_plot(single_file):
    num_spheres = 25

    xposdata, yposdata, xfftmatrix, yfftmatrix, frequency_bins, fs = hdf5_sphere_data_scraper(single_file)
    print(fs)
    inc = 1/fs
    t = np.arange(0,len(xposdata[:,0]),1)*inc
    fig, ax = plt.subplots(5,5, tight_layout=True)
    figpos, axpos = plt.subplots(2,1, sharex=True)

    sphrnum = 0
    for i in range(5):
        for j in range(5):
            label_name = "Sphere " + str(sphrnum)

            ax[i][j].semilogy(frequency_bins, xfftmatrix[:,sphrnum])
            ax[i][j].semilogy(frequency_bins, yfftmatrix[:,sphrnum])
            ax[i][j].set_xlim([5,250])
            ax[i][j].set_xlabel('Frequency (Hz)')
            ax[i][j].set_ylabel(r'ASD ($m/ \sqrt{Hz}$)')
            ax[i][j].set_title(label_name)

            axpos[0].plot(t,xposdata[:,sphrnum], label=label_name)
            axpos[1].plot(t,yposdata[:,sphrnum], label=label_name)
            sphrnum += 1

    ax[0][4].legend(['X motion','Y motion'], loc = 'lower right', bbox_to_anchor=(1, 1.3), ncol=1)

    handles, labels = axpos[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axpos[0].legend(by_label.values(), by_label.keys(), fontsize=12, loc="upper right", bbox_to_anchor=(1.1, 1), borderaxespad=0)

    fig.suptitle(r"Amplitude Spectral Densities of a 56 $\mu$m spaced 5 x 5 Array of 10 um Spheres at 0.4 mbar")
    axpos[0].set_xlim([0,max(t)])
    axpos[1].set_xlim([0,max(t)])
    axpos[0].set_ylabel("Position in X Direction (m)")
    axpos[1].set_ylabel("Position in Y Direction (m)")
    axpos[1].set_xlabel('Time (s)')
    figpos.suptitle(r"Timestream Position Data for a 56 $\mu$m spaced 5 x 5 Array of 10 um Spheres at 0.4 mbar")
    plt.show()



folder_list = ['0-8MHz', '1MHz', '1-25MHz', '1-5MHz']

single_file = r'D:\Lab data\20240604\0-8MHz\0-8MHz_25grid-lp-1.h5'
array_single_file_plot(single_file)