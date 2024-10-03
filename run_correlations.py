'''
Correlation calculator running script
'''
import os
import numpy as np
import scipy
import matplotlib as mpl 
import matplotlib.pyplot as plt
from scipy import fft
from sklearn.decomposition import PCA ## need this for principle component analysis below
from matplotlib.mlab import psd
from scipy.optimize import minimize, curve_fit
from scipy.stats import chi2, pearsonr
from scipy.signal import welch
from scipy.signal.windows import blackman
from scipy.signal import find_peaks, butter, lfilter, csd, coherence
from matplotlib.pyplot import gca
import h5py
import pandas as pd
from matplotlib.colors import LogNorm
import seaborn as sn
from scipy.interpolate import interp1d
import matplotlib.pylab as pl
import matplotlib.gridspec as gridspec

from correlationcalculator import *


main_directory = r"D:\Lab data\20240905\hdf5_datafiles"
totalspheres = 2
saveflag = False
savefigs = False
anticrossinglbs = [150,120]
anticrossingubs = [225,190]

color_value = np.linspace(0,1,totalspheres)
color_value_T = color_value[::-1]
color_codes = [(color_value[i],0,color_value_T[i]) for i in range(totalspheres)]

colorvalue1 = np.linspace(0,0.8,12)
colorvalue2 = np.linspace(0.55,1,12)
viridis = mpl.colormaps['viridis'].resampled(20)
inferno = mpl.colormaps['inferno'].resampled(20)
colormap1 = [viridis(colorvalue1[i]) for i in range(10)]
colormap2 = [inferno(colorvalue2[i]) for i in range(10)]

jointfig = plt.figure()
grid = plt.GridSpec(11, 10, wspace=3.5, hspace=4)
sph0 = jointfig.add_subplot(grid[:4, :7])
sph1 = jointfig.add_subplot(grid[5:9, :7])
jointcor = jointfig.add_subplot(grid[:, 7:])

jointcor.tick_params(labelsize=16)
sph0.tick_params(labelsize=16)
sph1.tick_params(labelsize=16)
jointcor.set_title('Correlation vs Separation',fontsize = 26)
jointcor.set_xlabel(r'Separation ($\mu m$)',fontsize=20)
jointcor.set_ylabel('Pearson Correlation Coefficient',fontsize=20)

x_peak_scan, y_peak_scan, separation_scan, correlation_scan, freqasddata, xasddata, yasddata, xcross_SD_list, ycross_SD_list, coherfreq = folder_walker_correlation_calc(main_directory, totalspheres, saveflag, savefigs)
jointcor = plot_correlations_vs_separations(x_peak_scan, y_peak_scan, separation_scan, correlation_scan, main_directory, totalspheres, savefigs, color_codes, jointcor)
[sph0, sph1] =plot_separation_ASD_scan(freqasddata, xasddata, yasddata, separation_scan, main_directory, savefigs, colormap1,colormap2, [sph0, sph1])
heatmap_scan_plotter(freqasddata, xasddata, yasddata,  anticrossinglbs, anticrossingubs, separation_scan, main_directory, totalspheres, savefigs)

plt.show()