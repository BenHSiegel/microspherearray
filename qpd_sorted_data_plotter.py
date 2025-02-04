'''
Takes a folder filled with hdf5 data files from the QPD
Plots the PSDs of the spheres averaged over the files
Right now only takes presorted data
'''

import h5py
import numpy as np
import scipy
from scipy.signal import welch
import matplotlib
from matplotlib import pyplot as plt
import os

def hdf5_scraper(filename):
    '''
    Basic hdf5 reader for the qpd sorted data files.
    filename = path to the file you want to read
    Outputs:

    xfftmatrix = numpy array where each column in the array is the x amplitude spectral density data from a sphere (in m/root(Hz))
    yfftmatrix = numpy array where each column in the array is the y amplitude spectral density data from a sphere (in m/root(Hz))
    frequency_bins = 1D array containing the frequency bins that were used in the PSD calculations
    '''
    #Opens the HDF5 file in read mode  
    hf = h5py.File(filename, 'r')

    sum_data= hf.get('SUM')
    x_data= hf.get('X')
    y_data= hf.get('Y')
    sampleT = hf.attrs['Fsamp']
    framerate = 1/sampleT
    xdata = np.array(x_data[0,:])
    ydata = np.array(y_data[0,:])
    zdata = np.array(sum_data[0,:])
    time = (np.arange(len(xdata))) * sampleT
    hf.close()

    return xdata, ydata, zdata, framerate

#def asd_calc(xdata,ydata,zdata,fs):

filename = r'D:\Lab data\20250124\test1_11am\QPD\nofeedback_1-5mbar_beamsorted_0.h5'
x, y, z, fs = hdf5_scraper(filename)
print(x)
print(fs)

segmentsize = len(x)
xfreq, xPSD = welch(x, fs, 'hann', segmentsize)  #, segmentsize/2, fftbinning, 'constant', True, 'density', 0,'mean'
xASD = np.sqrt(xPSD)

yfreq, yPSD = welch(y, fs, 'hann', segmentsize)  #, segmentsize/2, fftbinning, 'constant', True, 'density', 0,'mean'
yASD = np.sqrt(yPSD)

zfreq, zPSD = welch(z, fs, 'hann', segmentsize)  #, segmentsize/2, fftbinning, 'constant', True, 'density', 0,'mean'
zASD = np.sqrt(zPSD)


fig, ax = plt.subplots(3,1)
ax[0].loglog(xfreq,xASD)
ax[2].loglog(zfreq,zASD)
ax[1].loglog(yfreq,yASD)

plt.show()