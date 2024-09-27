# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 19:02:05 2024

@author: Ben
"""
import pandas as pd
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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

filenamelist = [r"D:\Lab data\20240919\QPD\4beams3trapped_AIstream_0.h5",r"D:\Lab data\20240919\QPD\4beams3trapped_AIstream_1.h5",r"D:\Lab data\20240919\QPD\4beams3trapped_AIstream_2.h5",r"D:\Lab data\20240919\QPD\4beams3trapped_AIstream_3.h5",r"D:\Lab data\20240919\QPD\4beams3trapped_AIstream_4.h5"]
counter=0
for filename in filenamelist:
    hf = h5py.File(filename, 'r')
    position_data= hf.get('unsorted_pos_data')

    sampleT = hf.attrs['Fsamp']
    CH0 = np.array(position_data[0,:])
    CH1 = np.array(position_data[1,:])
    CH2 = np.array(position_data[2,:])
    trig = np.array(position_data[3,:])
    time = (np.arange(len(CH0))+counter*len(CH0)) * sampleT
    hf.close()


    if counter ==0:
        df = pd.DataFrame({'Time': time, 'CH0': CH0, 'CH1': CH1, 'CH2': CH2, 'trig':trig})
        triglocs = []
        for i in range(len(trig)):
            if trig[i] != trig[i-1]:
                triglocs.append(i+38)
    else:
        
        newdf = pd.DataFrame({'Time': time, 'CH0': CH0, 'CH1': CH1, 'CH2': CH2, 'trig':trig})
        df = pd.concat([df,newdf],ignore_index=True)

    counter +=1



fig1, ax = plt.subplots()

ax.plot(df.Time,df.CH0)
ax.plot(df.Time,df.CH1)
ax.plot(df.Time,df.CH2)
#ax.plot(df.Time,df.trig)
ax2 = inset_axes(ax, width='40%', height='50%', loc=1)

ax2.tick_params(axis='both', which='major', labelsize=18)
ax2.set_xlim(df.Time[3000],df.Time[7000])
ax2.plot(df.Time[3000:7000],df.CH0[3000:7000], linewidth=2.5)
ax2.plot(df.Time[3000:7000],df.CH1[3000:7000], linewidth=2.5)
ax2.plot(df.Time[3000:7000],df.CH2[3000:7000], linewidth=2.5)
#ax2.plot(df.Time*1E6,df.trig)
y = np.arange(-1,1,0.001)
for i in range(len(triglocs)-1):
    
    if 2000<triglocs[i]<7500:
        ax2.vlines(df.Time[triglocs[i]],-1, 1, colors = 'r', linestyles = 'dashed', linewidths = 1)
        ax2.fill_betweenx(y, df.Time[triglocs[i]] + 60E-6, df.Time[triglocs[i+1]]-1E-7, facecolor='#aaf0a1',alpha=0.3)
ax.legend(['QPD X','QPD Y', 'QPD Sum'],loc = 'upper left',fontsize=20, framealpha = 1)
ax.set_xlabel(r'Time ($\mu$s)',fontsize=28)
ax.set_ylabel('QPD Response (V)',fontsize=28)
#ax2.set_title('Timestream of Quadrant Photodiode for 4 Spheres',fontsize=28,pad=15)
ax2.set_ylim(-0.75,0.5)
plt.show()