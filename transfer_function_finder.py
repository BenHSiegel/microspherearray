'''
Scrapes hdf5 files to find a transfer function
'''

import numpy as np
import h5py
import scipy as sp
import matplotlib as mp
import matplotlib.pyplot as plt
from scipy.signal import welch

import basichdf5
import os
from collections import defaultdict

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
    print(xdata.shape)
    hf.close()

    return xdata, ydata, zdata, framerate


def file_psd_averager(folder, filelist):
    counter = 0
    for i in filelist:
        xdata, ydata, zdata, framerate = hdf5_scraper(os.path.join(folder,i))
        if len(xdata.shape) > 1:
            totalspheres = xdata.shape[0]
        else:
            totalspheres = 1
        
        if counter == 0:
            xpsd_matrix = [[] for j in range(totalspheres)]
            ypsd_matrix = [[] for j in range(totalspheres)]
            zpsd_matrix = [[] for j in range(totalspheres)]
            
            xpsd_avg = [[] for j in range(totalspheres)]
            ypsd_avg = [[] for j in range(totalspheres)]
            zpsd_avg = [[] for j in range(totalspheres)]

            
        for k in range(totalspheres):
            xfreq, xPSD = welch(xdata, framerate)
            yfreq, yPSD = welch(ydata, framerate)
            zfreq, zPSD = welch(zdata, framerate)
            if counter == 0:
                xpsd_matrix[k] = xPSD.reshape(-1,1)
                ypsd_matrix[k] = yPSD.reshape(-1,1)
                zpsd_matrix[k] = zPSD.reshape(-1,1)
            else:
                xpsd_matrix[k] = np.concatenate((xpsd_matrix[k], xPSD.reshape(-1,1)), axis = 1)
                ypsd_matrix[k] = np.concatenate((ypsd_matrix[k], yPSD.reshape(-1,1)), axis = 1)
                zpsd_matrix[k] = np.concatenate((zpsd_matrix[k], zPSD.reshape(-1,1)), axis = 1)
        
        counter += 1
    
    for i in range(totalspheres):
        xpsd_avg[i] = np.mean(xpsd_matrix[i], axis=1)
        ypsd_avg[i] = np.mean(ypsd_matrix[i], axis=1)
        zpsd_avg[i] = np.mean(zpsd_matrix[i], axis=1)
    
    return xpsd_avg, ypsd_avg, zpsd_avg, xfreq


def folder_sorting(directory):
    groups = defaultdict(list)
    
    for filename in os.listdir(directory):
        basename, extension = os.path.splitext(filename)
        if extension == '.h5':
            settings, session = basename.split('_beamsorted_')
            groups[settings].append(filename)
            

    settings_list = [i for i in groups if groups[i]!=groups.default_factory()]
    
    return groups, settings_list


xmod = r'D:\Lab data\20250214\CH0 modulate'

static = r'D:\Lab data\20250214\No modulate'

static_group, static_setting = folder_sorting(static)

for i in static_setting:
    xref, yref, zref, freqref = file_psd_averager(static, static_group[i])

groups, settings_list = folder_sorting(xmod)

figx, axx = plt.subplots()
figy, axy = plt.subplots()
figz, axz = plt.subplots()

for i in settings_list:
    xpsd, ypsd, zpsd, freq = file_psd_averager(xmod, groups[i])
    x_comp = np.subtract(xpsd,xref)
    y_comp = np.subtract(ypsd,yref)
    z_comp = np.subtract(zpsd,zref)
    axx.loglog(freq, x_comp[0], label=i)
    axy.loglog(freq, y_comp[0], label=i)
    axz.loglog(freq, z_comp[0], label=i)

axx.set_xlabel('Frequency (Hz)')
axy.set_xlabel('Frequency (Hz)')
axz.set_xlabel('Frequency (Hz)')

axx.set_ylabel('PSD (V^2/Hz)')
axy.set_ylabel('PSD (V^2/Hz)')
axz.set_ylabel('PSD (V^2/Hz)')

axx.set_title('X Motion')
axy.set_title('Y Motion')
axz.set_title('Z Motion')

axx.legend()
axy.legend()
axz.legend()

plt.show()