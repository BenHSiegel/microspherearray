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
param_scan_values = ['140 um', '105 um', '70 um']

sphere_peak_list = [[] for i in range(num_spheres)]


for file in file_list:
    hf = h5py.File(file, 'r')
    
    freqs = np.array(hf.get('frequencies'))
    X_asd = np.array(hf.get('XASD RMS Avg'))
    Y_asd = np.array(hf.get('YASD RMS Avg'))
    
    
    x_peaks_list = [[] for i in range(len(X_asd))]
    y_peaks_list = [[] for i in range(len(X_asd))]
    
    
    for i in range(len(X_asd)):
        x_peak_indices, x_peak_dict = find_peaks(X_asd[i,:], height=3E-9, threshold=1E-11)
        x_peak_heights = x_peak_dict['peak_heights']
        x_peak_freqs = freqs[x_peak_indices]
        
        x_peaks = (np.vstack((x_peak_freqs, x_peak_heights))).T
        x_peaks_list[i] = x_peaks
        
        y_peak_indices, y_peak_dict = find_peaks(Y_asd[i,:], height=3E-9, threshold=1E-11)
        y_peak_heights = y_peak_dict['peak_heights']
        y_peak_freqs = freqs[y_peak_indices]
        
        y_peaks = (np.vstack((y_peak_freqs, y_peak_heights))).T
        y_peaks_list[i] = y_peaks
        
        if i < num_spheres:
            sphere_peak_list[i].append([x_peaks, y_peaks])


figs= {}
axs = {} 
figy = {}
ays = {}  
plt.rcParams.update({'font.size': 12}) 
for i in range(len(sphere_peak_list)):
    figs[i], axs[i] = plt.subplots(2, 1, sharex=True, tight_layout=True)
    figy[i], ays[i] = plt.subplots(2, 1, sharex=True, tight_layout=True)
    
    figs[i].set_size_inches(8.5, 11)
    figs[i].set_dpi(600) 
    
    figy[i].set_size_inches(8.5, 11)
    figy[i].set_dpi(600)
    
    sphere = sphere_peak_list[i]
    xmaxlen = max(len(f[0]) for f in sphere)
    ymaxlen = max(len(f[1]) for f in sphere)
    xcomp = []
    ycomp = []
    
    for f in sphere:
        if len(f[0]) == xmaxlen:
            xcomp = f[0]
            
        if len(f[1]) == ymaxlen:
            ycomp = f[1]
    
            
    
    for j in xcomp[:,0]:
        
        complistfq = []
        complistamp = []
        complist = []
        for f in range(len(sphere)):
            for k in range(len(sphere[f][0][:,0])):
                peak = sphere[f][0][k,0]
                amp = sphere[f][0][k,1]
                if ((j - 7) <= peak <= (j + 7)):
                    complistfq.append(peak)
                    complistamp.append(amp)
                    complist.append(f)
        
        if len(complist) > 1:
            param_values = [param_scan_values[i] for i in complist]
            difffqs = (complistfq-complistfq[0])
            diffamps = (complistamp - complistamp[0])/complistamp[0] * 100
            
            axs[i][0].scatter(param_values, difffqs, label=(str(round(complistfq[0],2)) + ' Hz'))
            axs[i][0].plot(param_values, difffqs, '-.')   
            
            axs[i][1].scatter(param_values, diffamps, label=(str(round(complistfq[0],2)) + ' Hz'))
            axs[i][1].plot(param_values, diffamps, '-.')   
    
    
    
    axs[i][1].tick_params(axis='x', labelrotation=-45)
    axs[i][1].set_xlabel('Separation')
    axs[i][0].set_ylabel('Frequency shift [Hz]')
    axs[i][1].set_ylabel('Amplitude Difference [%]')
    axs[i][0].legend(bbox_to_anchor=(1, 1.01), loc=4, borderaxespad=0.)
    
    axs[i][0].set_title("Evolution of X Motion for Sphere " + str(i))
    
    for j in ycomp[:,0]:
        
        complistfq = []
        complistamp = []
        complist = []
        for f in range(len(sphere)):
            for k in range(len(sphere[f][1][:,0])):
                peak = sphere[f][1][k,0]
                amp = sphere[f][1][k,1]
                if ((j - 5) <= peak <= (j + 5)):
                    complistfq.append(peak)
                    complistamp.append(amp)
                    complist.append(f)
        
        if len(complist) > 1:
            param_values = [param_scan_values[i] for i in complist]
            difffqs = (complistfq-complistfq[0])
            diffamps = (complistamp - complistamp[0])/complistamp[0] * 100
            
            ays[i][0].scatter(param_values, difffqs, label=(str(round(complistfq[0],2)) + ' Hz'))
            ays[i][0].plot(param_values, difffqs, '-.')   
            
            ays[i][1].scatter(param_values, diffamps, label=(str(round(complistfq[0],2)) + ' Hz'))
            ays[i][1].plot(param_values, diffamps, '-.')   
    
    
    
    ays[i][1].tick_params(axis='x', labelrotation=-45)
    ays[i][1].set_xlabel('Separation')
    ays[i][0].set_ylabel('Frequency shift [Hz]')
    ays[i][1].set_ylabel('Amplitude Difference [%]')
    ays[i][0].legend(bbox_to_anchor=(1, 1.01), loc=4, borderaxespad=0.)
    
    ays[i][0].set_title("Evolution of Y Motion for Sphere " + str(i))

    
    # highest_peak_index = peak_indices[np.argmax(peak_heights)]
    # second_highest_peak_index = peak_indices[np.argpartition(peak_heights,-2)[-2]]

# figs={}
# axs={}
# for i in range(len(sphere_peak_list)):
    
#     figs[i], axs[i] = plt.subplots(2, 1, sharex=True, tight_layout=True)
    
#     figs[i].set_size_inches(8.5, 11)
#     figs[i].set_dpi(600)
#     plt.rcParams.update({'font.size': 12})
    
#     j=0
#     for filedata in sphere_peak_list[i]:
#         axs[i][0].scatter(filedata[0][:,0], filedata[0][:,1], label =(param_scan_values[j]))
#         axs[i][1].scatter(filedata[1][:,0], filedata[1][:,1], label =(param_scan_values[j]))

#         j += 1
        
#     figs[i].suptitle("Sphere " + str(i) + ' peak responce for varied distance')
    
#     #axs[i][0].set_xlabel('Frequency [Hz]')
#     axs[i][0].set_ylabel(r'ASD [$m/ \sqrt{Hz}$]')
#     axs[i][0].set_title('X ASD')
#     axs[i][0].legend(bbox_to_anchor=(1, 1.01), loc=4, borderaxespad=0.)
    
#     axs[i][1].set_xlabel('Frequency [Hz]')
#     axs[i][1].set_ylabel(r'ASD [$m/ \sqrt{Hz}$]')
#     axs[i][1].set_title('Y ASD')
#     #axs[i][1].legend()


 
        
        

        