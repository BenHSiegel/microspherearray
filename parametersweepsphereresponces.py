# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 11:43:57 2024

@author: Ben
"""
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from matplotlib.pyplot import gca
import h5py

file_list = []

for file in file_list:
    hf = h5py.File(file, 'r')
    
    freqs = np.array(hf.get('frequencies'))
    X_asd = np.array(hf.get('XASD RMS Avg'))
    Y_asd = np.array(hf.get('YASD RMS Avg'))
    
    
    x_peaks_list = [[] for i in range(len(X_asd))]
    y_peaks_list = [[] for i in range(len(X_asd))]
    
    
    for i in range(len(X_asd)):
        x_peak_indices, x_peak_dict = find_peaks(X_asd[i,:], height=2E-9)
        x_peak_heights = x_peak_dict['peak_heights']
        x_peak_freqs = freqs[x_peak_indices]
        
        x_peaks = (np.vstack((x_peak_indices, x_peak_freqs, x_peak_heights))).T
        x_peaks_list[i] = x_peaks
        
        y_peak_indices, y_peak_dict = find_peaks(Y_asd[i,:], height=2E-9)
        y_peak_heights = y_peak_dict['peak_heights']
        y_peak_freqs = freqs[y_peak_indices]
        
        y_peaks = (np.vstack((y_peak_indices, y_peak_freqs, y_peak_heights))).T
        y_peaks_list[i] = y_peaks
        

    # highest_peak_index = peak_indices[np.argmax(peak_heights)]
    # second_highest_peak_index = peak_indices[np.argpartition(peak_heights,-2)[-2]]
    
    for i in range(len(x_peaks_list)):
        for j in range(len(x_peaks_list[i])):
            X_asdi = X_asd[i]
            lb = int(x_peaks_list[i][j,0]-6)
            up = int(x_peaks_list[i][j,0]+10)


        
        
        
        