# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 11:43:57 2024

@author: Ben
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from matplotlib.pyplot import gca
import h5py

path = r"C:\Users\bensi\Documents\Research\20240319\middle"
filename = '\middle0-6mbar_vid4.avi'

fullpath = path + filename

hf = h5py.File(fullpath, 'r')
freqs = hf.frequencies
X_psd = hf.X_psd
Y_psd = hf.Y_psd

# peak_indices, peak_dict = signal.find_peaks(spline, height=(None, None), distance=50)
# peak_heights = peak_dict['peak_heights']
# highest_peak_index = peak_indices[np.argmax(peak_heights)]
# second_highest_peak_index = peak_indices[np.argpartition(peak_heights,-2)[-2]]