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


def justpeakplotter(path, color_codes):
    hf = h5py.File(path, 'r')
    freqs = np.array(hf.get('frequencies'))
    X_asd = np.array(hf.get('XASD RMS Avg'))
    Y_asd = np.array(hf.get('YASD RMS Avg'))
    
    #mpl.rcParams['figure.dpi'] = 1200
    
    x_peaks_list = [[] for i in range(len(X_asd))]
    y_peaks_list = [[] for i in range(len(X_asd))]
    
    fig0, ax0 = plt.subplots()
    fig1, ax1 = plt.subplots()
    legendlist = []
    offset = np.linspace(0,5E-15,len(X_asd))
    offsety = np.linspace(0,3E-14,len(X_asd))
    for i in range(len(X_asd)):
        ax0.plot(freqs, (X_asd[i,:]**2 + offset[i]), color=color_codes[i])
        x_peak_indices, x_peak_dict = find_peaks(X_asd[i,:], height=3E-9)
        x_peak_heights = x_peak_dict['peak_heights']
        x_peak_freqs = freqs[x_peak_indices]
        
        x_peaks = (np.vstack((x_peak_indices, x_peak_freqs, x_peak_heights))).T

        
        x_peaks_list[i] = x_peaks
        
        ax1.plot(freqs, Y_asd[i,:]**2 + offsety[i], color=color_codes[i])
        y_peak_indices, y_peak_dict = find_peaks(Y_asd[i,:], height=3E-9, threshold=1E-11)
        y_peak_heights = y_peak_dict['peak_heights']
        y_peak_freqs = freqs[y_peak_indices]
        
        y_peaks = (np.vstack((y_peak_indices, y_peak_freqs, y_peak_heights))).T
        y_peaks_list[i] = y_peaks
        
        legendlist.append('Sphere ' + str(i))
    
    ax0.set_xlim(60,200)
    ax0.set_ylim(0,6.5E-15)
    ax0.set_xlabel('Frequency (Hz)', fontsize=18)
    ax0.set_ylabel('Sphere Index', fontsize=18)
    spherenames = [str(x+1) for x in range(25)]
    ax0.set_yticks(offset, labels=spherenames)
    #ax0.legend(legendlist, fontsize=12, bbox_to_anchor=(1.04, 0), loc="lower left", borderaxespad=0)
    ax0.set_title('X Motion Spectra', fontsize=22)
    
    ax1.set_xlim(60,150)
    ax1.set_ylim(0,3.2E-14)
    ax1.set_xlabel('Frequency (Hz)', fontsize=18)
    ax1.set_ylabel('Sphere Index', fontsize=18)
    ax1.set_yticks(offsety, labels=spherenames)
    #ax1.legend(legendlist, fontsize=12, bbox_to_anchor=(1.04, 0), loc="lower left", borderaxespad=0)
    ax1.set_title('Y Motion Spectra', fontsize=22)
        
    # highest_peak_index = peak_indices[np.argmax(peak_heights)]
    # second_highest_peak_index = peak_indices[np.argpartition(peak_heights,-2)[-2]]
    
    fig, ax = plt.subplots()
    
    for i in range(len(x_peaks_list)):
        sortedpeaksind = np.argsort(x_peaks_list[i][:,2])[::-1]
        numpeaks = 2
        counter = 0
        j=0
        while counter < numpeaks:
            peakind = sortedpeaksind[j]
            if x_peaks_list[i][peakind,1] > 60:
                X_asdi = X_asd[i]
                lb = int(x_peaks_list[i][peakind,0]-10)
                up = int(x_peaks_list[i][peakind,0]+10)
                ax.semilogy(freqs[lb:up], X_asdi[lb:up], color=color_codes[i], label =('Sphere ' + str(i+1)))
                
                counter+=1
            j+=1

                
    ax.grid()
    #ax.set_xlim(0,170)
    ax.set_xlabel('Frequency (Hz)', fontsize=18)
    ax.set_ylabel(r'ASD ($m/ \sqrt{Hz}$)', fontsize=18)
    
    handles, labels = gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=12, bbox_to_anchor=(1.04, 0), loc="lower left", borderaxespad=0)
    
    ax.set_title('X motion ASD', fontsize=22)
    
    
    figa, axa = plt.subplots()
    
    for i in range(len(y_peaks_list)):
        sortedpeaksind = np.argsort(y_peaks_list[i][:,2])[::-1]
        numpeaks = 2
        counter = 0
        j=0
        while counter < numpeaks:
            if j < len(sortedpeaksind):
                peakind = sortedpeaksind[j]
                if y_peaks_list[i][peakind,1] > 60:
                    Y_asdi = Y_asd[i]
                    lb = int(y_peaks_list[i][peakind,0]-10)
                    up = int(y_peaks_list[i][peakind,0]+10)
                    axa.semilogy(freqs[lb:up], Y_asdi[lb:up], color=color_codes[i], label =('Sphere ' + str(i+1)))
                    
                    counter+=1
                j+=1
            else:
                break
                
    axa.grid()
    #axa.set_xlim(0,170)
    axa.set_xlabel('Frequency (Hz)', fontsize=18)
    axa.set_ylabel(r'ASD ($m/ \sqrt{Hz}$)', fontsize=18)
    
    handles, labels = gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axa.legend(by_label.values(), by_label.keys(), fontsize=12, bbox_to_anchor=(1.04, 0), loc="lower left", borderaxespad=0)
    
    axa.set_title('Y motion ASD', fontsize=22)

    plt.show()
    
###############################################################################
    
totalspheres = 25
file = r'D:\Lab data\20240604\0-8MHz\0-8MHz_rmsavg.h5'
color_map = mpl.cm.get_cmap('viridis')
colorind = np.linspace(0,0.7,totalspheres)
color_codes = [color_map(colorind[i]) for i in range(totalspheres)]

justpeakplotter(file, color_codes)