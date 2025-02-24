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
    segmentsize = 2048
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
            xfreq, xPSD = welch(xdata, framerate, 'hann', segmentsize)
            yfreq, yPSD = welch(ydata, framerate, 'hann', segmentsize)
            zfreq, zPSD = welch(zdata, framerate, 'hann', segmentsize)
            if counter == 0:
                xfreq = xfreq.reshape(1,-1)
                xpsd_matrix[k] = xPSD.reshape(-1,1)
                ypsd_matrix[k] = yPSD.reshape(-1,1)
                zpsd_matrix[k] = zPSD.reshape(-1,1)
            else:
                xfreq = xfreq.reshape(1,-1)
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


filepath = r'D:\Lab data\20250219'

groups, settings_list = folder_sorting(filepath)

conditionslist = ['0.1 mbar', '0.015 mbar', '2.8 mbar shaking', '2.8 mbar', '0.4 mbar', '0.77 mbar shaking','0.77 mbar', 'No sphere']

figs = {}
axs = {}

for i in range(len(settings_list)):
    xpsd, ypsd, zpsd, freq = file_psd_averager(filepath, groups[settings_list[i]])
    
    figs[i], axs[i] = plt.subplots(3, 1, sharex=True, tight_layout=True)
    print(freq)
    print(xpsd)

    axs[i][0].semilogy(freq[0,2:150], xpsd[0][2:150])
    axs[i][1].semilogy(freq[0,2:150], ypsd[0][2:150])
    axs[i][2].semilogy(freq[0,2:150], zpsd[0][2:150])
    axs[i][0].set_xlim(5,500)
    axs[i][1].set_xlim(5,500)
    axs[i][2].set_xlim(5,500)
    axs[i][0].set_ylabel('X PSD (V^2/Hz)')
    axs[i][1].set_ylabel('Y PSD (V^2/Hz)')
    axs[i][2].set_ylabel('Z PSD (V^2/Hz)')
    axs[i][2].set_xlabel('Frequency (Hz)')
    figs[i].suptitle(conditionslist[i])


plt.show()