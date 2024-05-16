# -*- coding: utf-8 -*-
"""
Created on Thu May  9 14:29:27 2024

@author: yalem
"""
import av as av
import pims
import trackpy as tp
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import fft
from sklearn.decomposition import PCA ## need this for principle component analysis below
from matplotlib.mlab import psd
from scipy.optimize import minimize, curve_fit
from scipy.stats import chi2
import os
from scipy.signal import welch
from scipy.signal.windows import blackman
from scipy.signal import find_peaks
from matplotlib.pyplot import gca
import h5py

from folderscanning_videoprocessor import *



path = r"D:\Lab data\20240509\3MHzline"
os.chdir(path)
framerate = 700


## This part is used to run on the first video and make sure you have the right
## minmass value for processmovie

# filename = "3MHzsep-1.avi"
# [spheres, f] = processmovie(filename, framerate)
# fig, ax = plt.subplots()
# ax.hist(f['mass'], bins=1000)



## This part will run through a single folder, processing the avi in it and
## computing the RMS ASDs from them all

pcacheck = False
saveposdata = True
saveFFTavg = True
fftsave = "3MHzline_rmsavg"

totalspheres = videofolder_dataextractions(path, framerate, pcacheck, saveposdata)
hdf5file_RMSprocessing(path, totalspheres, saveFFTavg, fftsave)