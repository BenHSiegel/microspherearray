'''
Tracker comparison plot maker
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
from scipy.ndimage import variance
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
from basichdf5 import hdf5_sphere_data_scraper as scraper


path = r'C:\Users\Ben\Documents\Research\Tracker comparison'
os.chdir(path)
_, _, ASDx_trackpy, ASDy_trackpy, f_trackpy, framerate = scraper('ulp1.h5')

cnn = np.load('cnn_translations_ulp1.npy')
eigenframe = np.load('eigenframe_translations_ulp1.npy')

fftbinning = 2048
segmentsize = round(framerate/4)
fx_eigen, PSDx_eigen = welch(eigenframe[:,0], framerate, 'hann', segmentsize, segmentsize/2, fftbinning, 'constant', True, 'density', 0,'mean')
fy_eigen, PSDy_eigen = welch(eigenframe[:,1], framerate, 'hann', segmentsize, segmentsize/2, fftbinning, 'constant', True, 'density', 0,'mean')
fx_cnn, PSDx_cnn = welch(cnn[:,0], framerate, 'hann', segmentsize, segmentsize/2, fftbinning, 'constant', True, 'density', 0,'mean')
fy_cnn, PSDy_cnn = welch(cnn[:,1], framerate, 'hann', segmentsize, segmentsize/2, fftbinning, 'constant', True, 'density', 0,'mean')
ASDx_eigen = np.sqrt(PSDx_eigen)
ASDy_eigen = np.sqrt(PSDy_eigen)
ASDx_cnn = np.sqrt(PSDx_cnn)
ASDy_cnn = np.sqrt(PSDy_cnn)


fig, ax = plt.subplots()
ax.semilogy(f_trackpy[:-1], ASDx_trackpy[:-1],label='TrackPy', linewidth=2)
ax.semilogy(fx_eigen[:-1], ASDx_eigen[:-1],label='Eigenframe',linewidth=2)
ax.semilogy(fx_cnn[:-1], ASDx_cnn[:-1],label='CNN',linewidth=2)

ax.legend()
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel(r'Amplitude Spectral Density (m/' + u"\u221A" +'Hz)')

fig1, ax1 = plt.subplots()

ax1.loglog(f_trackpy[:-1], ASDx_trackpy[:-1],label='TrackPy',linewidth=3)
ax1.loglog(fx_eigen[:-1], ASDx_eigen[:-1],label='Eigenframe',linewidth=3)
ax1.loglog(fx_cnn[:-1], ASDx_cnn[:-1],label='CNN',linewidth=3)
ax1.set_xlim([10,500])
ax1.legend(fontsize=16,loc='lower left')
ax1.set_xlabel('Frequency (Hz)',fontsize=18)
ax1.set_ylabel(r'Amplitude Spectral Density (m/' + u"\u221A" +'Hz)',fontsize=18)
ax1.tick_params(labelsize=14)
plt.show()
