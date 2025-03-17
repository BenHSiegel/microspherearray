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

    xfeedback= hf.get('X Feedback')
    yfeedback= hf.get('Y Feedback')
    zfeedback= hf.get('Z Feedback')
    sampleT = hf.attrs['Fsamp']
    Fs = 1/sampleT
    xin = np.array(xfeedback[0,:])
    yin = np.array(yfeedback[0,:])
    zin = np.array(zfeedback[0,:])
    xout = np.array(xfeedback[1,:])
    yout = np.array(yfeedback[1,:])
    zout = np.array(zfeedback[1,:])
    time = (np.arange(len(xin))) * sampleT
    print(xin.shape)
    hf.close()

    return xin, xout, yin, yout, zin, zout, Fs


def file_psd_averager(folder, filelist):
    counter = 0
    segmentsize = 2048
    for i in filelist:
        xin, xout, yin, yout, zin, zout, framerate = hdf5_scraper(os.path.join(folder,i))

        totalspheres = 1
        
        if counter == 0:
            xinpsd_matrix = [[] for j in range(totalspheres)]
            yinpsd_matrix = [[] for j in range(totalspheres)]
            zinpsd_matrix = [[] for j in range(totalspheres)]
            
            xinpsd_avg = [[] for j in range(totalspheres)]
            yinpsd_avg = [[] for j in range(totalspheres)]
            zinpsd_avg = [[] for j in range(totalspheres)]

            xoutpsd_matrix = [[] for j in range(totalspheres)]
            youtpsd_matrix = [[] for j in range(totalspheres)]
            zoutpsd_matrix = [[] for j in range(totalspheres)]
            
            xoutpsd_avg = [[] for j in range(totalspheres)]
            youtpsd_avg = [[] for j in range(totalspheres)]
            zoutpsd_avg = [[] for j in range(totalspheres)]

            
        for k in range(totalspheres):
            xfreq, xPSD = welch(xin, framerate, 'hann', segmentsize)
            yfreq, yPSD = welch(yin, framerate, 'hann', segmentsize)
            zfreq, zPSD = welch(zin, framerate, 'hann', segmentsize)
            xfreqout, xPSDout = welch(xout, framerate, 'hann', segmentsize)
            yfreqout, yPSDout = welch(yout, framerate, 'hann', segmentsize)
            zfreqout, zPSDout = welch(zout, framerate, 'hann', segmentsize)
            if counter == 0:
                xfreq = xfreq.reshape(1,-1)
                xinpsd_matrix[k] = xPSD.reshape(-1,1)
                yinpsd_matrix[k] = yPSD.reshape(-1,1)
                zinpsd_matrix[k] = zPSD.reshape(-1,1)

                xoutpsd_matrix[k] = xPSDout.reshape(-1,1)
                youtpsd_matrix[k] = yPSDout.reshape(-1,1)
                zoutpsd_matrix[k] = zPSDout.reshape(-1,1)
            else:
                xfreq = xfreq.reshape(1,-1)
                xinpsd_matrix[k] = np.concatenate((xinpsd_matrix[k], xPSD.reshape(-1,1)), axis = 1)
                yinpsd_matrix[k] = np.concatenate((yinpsd_matrix[k], yPSD.reshape(-1,1)), axis = 1)
                zinpsd_matrix[k] = np.concatenate((zinpsd_matrix[k], zPSD.reshape(-1,1)), axis = 1)

                xoutpsd_matrix[k] = np.concatenate((xoutpsd_matrix[k], xPSDout.reshape(-1,1)), axis = 1)
                youtpsd_matrix[k] = np.concatenate((youtpsd_matrix[k], yPSDout.reshape(-1,1)), axis = 1)
                zoutpsd_matrix[k] = np.concatenate((zoutpsd_matrix[k], zPSDout.reshape(-1,1)), axis = 1)

        counter += 1
    
    for i in range(totalspheres):
        xinpsd_avg[i] = np.mean(xinpsd_matrix[i], axis=1)
        yinpsd_avg[i] = np.mean(yinpsd_matrix[i], axis=1)
        zinpsd_avg[i] = np.mean(zinpsd_matrix[i], axis=1)

        xoutpsd_avg[i] = np.mean(xoutpsd_matrix[i], axis=1)
        youtpsd_avg[i] = np.mean(youtpsd_matrix[i], axis=1)
        zoutpsd_avg[i] = np.mean(zoutpsd_matrix[i], axis=1)
    
    return xinpsd_avg, yinpsd_avg, zinpsd_avg, xoutpsd_avg, youtpsd_avg, zoutpsd_avg, xfreq
    

def folder_sorting(directory):
    testfreqs = defaultdict(list)
    
    for filename in os.listdir(directory):
        basename, extension = os.path.splitext(filename)
        if extension == '.h5':
            frequency, _ = basename.split('Hz')
            testfreqs[frequency].append(filename)
            

    freqs_list = [i for i in testfreqs if testfreqs[i]!=testfreqs.default_factory()]
    return testfreqs, freqs_list


filepath = r'D:\Lab data\20250313\p=0.1 tests'

testfreqs, freqs_list = folder_sorting(filepath)

conditionslist = [15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225, 240, 255, 270, 285, 300, 315, 330, 360, 390, 420, 450, 500, 600, 700, 800, 900, 1000, 1253, 1507, 1772, 2000]

figs = {}
axs = {}

for i in range(len(freqs_list)):


# for i in range(len(settings_list)):
#     xpsd, ypsd, zpsd, freq = file_psd_averager(filepath, groups[settings_list[i]])
    
#     figs[i], axs[i] = plt.subplots(3, 1, sharex=True, tight_layout=True)
#     print(freq)
#     print(xpsd)

#     axs[i][0].semilogy(freq[0,2:150], xpsd[0][2:150])
#     axs[i][1].semilogy(freq[0,2:150], ypsd[0][2:150])
#     axs[i][2].semilogy(freq[0,2:150], zpsd[0][2:150])
#     axs[i][0].set_xlim(5,500)
#     axs[i][1].set_xlim(5,500)
#     axs[i][2].set_xlim(5,500)
#     axs[i][0].set_ylabel('X PSD (V^2/Hz)')
#     axs[i][1].set_ylabel('Y PSD (V^2/Hz)')
#     axs[i][2].set_ylabel('Z PSD (V^2/Hz)')
#     axs[i][2].set_xlabel('Frequency (Hz)')
#     figs[i].suptitle(conditionslist[i])


plt.show()