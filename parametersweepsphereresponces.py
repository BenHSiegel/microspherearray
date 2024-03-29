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

file_list = [r"C:\Users\bensi\Documents\Research\20240319\expanded\expandedposition20240319rmsavg.h5",
             r"C:\Users\bensi\Documents\Research\20240319\middle\middleposition20240319rmsavg.h5",
             r"C:\Users\bensi\Documents\Research\20240319\condensed\condensedposition20240319rmsavg.h5"]
num_spheres = 10
param_scan_values = ['140 um separation', '105 um separation', '70 um separation']

sphere_peak_list = [[] for i in range(num_spheres)]


for file in file_list:
    hf = h5py.File(file, 'r')
    
    freqs = np.array(hf.get('frequencies'))
    X_asd = np.array(hf.get('XASD RMS Avg'))
    Y_asd = np.array(hf.get('YASD RMS Avg'))
    
    
    x_peaks_list = [[] for i in range(len(X_asd))]
    y_peaks_list = [[] for i in range(len(X_asd))]
    
    
    for i in range(len(X_asd)):
        x_peak_indices, x_peak_dict = find_peaks(X_asd[i,:], height=2E-9, threshold=1E-11)
        x_peak_heights = x_peak_dict['peak_heights']
        x_peak_freqs = freqs[x_peak_indices]
        
        x_peaks = (np.vstack((x_peak_freqs, x_peak_heights))).T
        x_peaks_list[i] = x_peaks
        
        y_peak_indices, y_peak_dict = find_peaks(Y_asd[i,:], height=2E-9, threshold=1E-11)
        y_peak_heights = y_peak_dict['peak_heights']
        y_peak_freqs = freqs[y_peak_indices]
        
        y_peaks = (np.vstack((y_peak_freqs, y_peak_heights))).T
        y_peaks_list[i] = y_peaks
        
        if i < num_spheres:
            sphere_peak_list[i].append([x_peaks, y_peaks])
    

    # highest_peak_index = peak_indices[np.argmax(peak_heights)]
    # second_highest_peak_index = peak_indices[np.argpartition(peak_heights,-2)[-2]]

figs={}
axs={}
for i in range(len(sphere_peak_list)):
    
    figs[i], axs[i] = plt.subplots(1, 2, sharey=False, tight_layout=True)
    
    figs[i].set_size_inches(18.5, 10.5)
    figs[i].set_dpi(1600)
    plt.rcParams.update({'font.size': 22})
    
    j=0
    for filedata in sphere_peak_list[i]:
        axs[i][0].scatter(filedata[0][:,0], filedata[0][:,1], label =(param_scan_values[j]))
        axs[i][1].scatter(filedata[1][:,0], filedata[1][:,1], label =(param_scan_values[j]))

        j += 1
        
    figs[i].suptitle("Sphere " + str(i) + ' peak responce for varied distance')
    
    axs[i][0].set_xlabel('Frequency [Hz]')
    axs[i][0].set_ylabel(r'ASD [$m/ \sqrt{Hz}$]')
    axs[i][0].set_title('X PSD')
    axs[i][0].legend()
    
    axs[i][1].set_xlabel('Frequency [Hz]')
    axs[i][1].set_ylabel(r'ASD [$m/ \sqrt{Hz}$]')
    axs[i][1].set_title('Y ASD')
    axs[i][1].legend()
        
        
        