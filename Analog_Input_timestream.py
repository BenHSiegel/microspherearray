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

filename = r"E:\Lab data\20240919\QPD\4beams3trapped_AIstream_0.h5"
hf = h5py.File(filename, 'r')
position_data= hf.get('unsorted_pos_data')

sampleT = hf.attrs['Fsamp']
CH0 = np.array(position_data[0,:])
CH1 = np.array(position_data[1,:])
CH2 = np.array(position_data[2,:])
trig = np.array(position_data[3,:])
time = np.arange(len(CH0)) * sampleT
hf.close()

df = pd.DataFrame({'Time': time, 'CH0': CH0, 'CH1': CH1, 'CH2': CH2, 'trig':trig})

triglocs = []
for i in range(len(trig)):
    if trig[i] != trig[i-1]:
        triglocs.append(i+38)

fig1, ax = plt.subplots()

ax.plot(df.Time,df.CH0)
ax.plot(df.Time,df.CH1)
ax.plot(df.Time,df.CH2)
ax.plot(df.Time,df.trig)


fig2, ax2 = plt.subplots()

ax2.tick_params(axis='both', which='major', labelsize=18)
ax2.set_xlim(0,df.Time[5000]*1E6)
ax2.plot(df.Time*1E6,df.CH0, linewidth=2.5)
ax2.plot(df.Time*1E6,df.CH1, linewidth=2.5)
ax2.plot(df.Time*1E6,df.CH2, linewidth=2.5)
#ax2.plot(df.Time*1E6,df.trig)
ax2.vlines(df.Time[triglocs]*1E6,-1, 1, colors = 'r', linestyles = 'dashed', linewidths = 1)
ax2.legend(['QPD X','QPD Y', 'QPD Sum'],loc = 'upper right',fontsize=20, framealpha = 1)
ax2.set_xlabel(r'Time ($\mu$s)',fontsize=28)
ax2.set_ylabel('QPD Response (V)',fontsize=28)
#ax2.set_title('Timestream of Quadrant Photodiode for 4 Spheres',fontsize=28,pad=15)
ax2.set_ylim(-0.75,0.5)
plt.show()