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
filename = r"\middleposition20240319rmsavg.h5"

fullpath = path + filename

hf = h5py.File(fullpath, 'r')
freqs = np.array(hf.get('frequencies'))
X_asd = np.array(hf.get('XASD RMS Avg'))
Y_asd = np.array(hf.get('YASD RMS Avg'))

mpl.rcParams['figure.dpi'] = 1200

x_peaks_list = [[] for i in range(len(X_asd))]
y_peaks_list = [[] for i in range(len(X_asd))]

fig0, ax0 = plt.subplots()
legendlist = []

for i in range(len(X_asd)):
    ax0.semilogy(freqs, X_asd[i,:])
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
    
    legendlist.append('Sphere ' + str(i))

ax0.grid()
ax0.set_xlim(0,170)
ax0.set_xlabel('Frequency [Hz]', fontsize=18)
ax0.set_ylabel(r'ASD [$m/ \sqrt{Hz}$]', fontsize=18)
ax0.legend(legendlist, fontsize=12, bbox_to_anchor=(1.04, 0), loc="lower left", borderaxespad=0)
ax0.set_title('X motion ASD', fontsize=22)
    
# highest_peak_index = peak_indices[np.argmax(peak_heights)]
# second_highest_peak_index = peak_indices[np.argpartition(peak_heights,-2)[-2]]

color_codes = ['#FF0000', '#00FF00', '#0000FF', '#FF8C00', '#00FFFF', '#FF00FF', '#800000', '#808000', '#800080', '#008080', '#008000', '#000080']
fig, ax = plt.subplots()

for i in range(len(x_peaks_list)):
    for j in range(len(x_peaks_list[i])):
        X_asdi = X_asd[i]
        lb = int(x_peaks_list[i][j,0]-6)
        up = int(x_peaks_list[i][j,0]+10)
        ax.semilogy(freqs[lb:up], X_asdi[lb:up], color=color_codes[i], label =('Sphere ' + str(i)))
            
ax.grid()
ax.set_xlim(0,170)
ax.set_xlabel('Frequency [Hz]', fontsize=18)
ax.set_ylabel(r'ASD [$m/ \sqrt{Hz}$]', fontsize=18)

handles, labels = gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), fontsize=12, bbox_to_anchor=(1.04, 0), loc="lower left", borderaxespad=0)

ax.set_title('X motion ASD', fontsize=22)