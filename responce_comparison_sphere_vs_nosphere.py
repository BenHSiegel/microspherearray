import numpy as np
import h5py
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.signal import welch, correlate, correlation_lags
import basichdf5
import math
import os
from collections import defaultdict

def find_nearest(array,value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def hdf5_read(filename):
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

    xdata = hf.get('X')
    ydata = hf.get('Y')
    zdata = hf.get('SUM')
    sampleT = hf.attrs['Fsamp']
    Fs = 1/sampleT
    x = np.array(xdata)
    y = np.array(ydata)
    z = np.array(zdata)

    time = (np.arange(len(x))) * sampleT
    hf.close()

    return x, y, z, Fs

def folder_sorting(directory):
    testfreqs = defaultdict(list)
    
    for filename in os.listdir(directory):
        basename, extension = os.path.splitext(filename)
        if extension == '.h5':
            frequency, _ = basename.split('Hz')
            testfreqs[frequency].append(filename)
            

    freqs_list = [i for i in testfreqs if testfreqs[i]!=testfreqs.default_factory()]
    return testfreqs, freqs_list


def file_frequency_space_TF(folder, filelist, drivingfreq):
    counter = 0
    for i in filelist:
        xin, yin, zin, framerate = hdf5_read(os.path.join(folder,i))
        totalspheres = 1
        if counter == 0:
            xinpsd_matrix = [[] for j in range(totalspheres)]
            yinpsd_matrix = [[] for j in range(totalspheres)]
            zinpsd_matrix = [[] for j in range(totalspheres)]
            
            xinpsd_avg = [[] for j in range(totalspheres)]
            yinpsd_avg = [[] for j in range(totalspheres)]
            zinpsd_avg = [[] for j in range(totalspheres)]

            xg = [[] for j in range(totalspheres)]
            yg = [[] for j in range(totalspheres)]
            zg = [[] for j in range(totalspheres)]
            
            
        for k in range(totalspheres):
            xfreq, xPSD = welch(xin, framerate, 'hann', len(xin))
            yfreq, yPSD = welch(yin, framerate, 'hann', len(xin))
            zfreq, zPSD = welch(zin, framerate, 'hann', len(xin))

            if counter == 0:
                xfreq = xfreq.reshape(1,-1)
                xinpsd_matrix[k] = xPSD.reshape(-1,1)
                yinpsd_matrix[k] = yPSD.reshape(-1,1)
                zinpsd_matrix[k] = zPSD.reshape(-1,1)


            else:
                xfreq = xfreq.reshape(1,-1)
                xinpsd_matrix[k] = np.concatenate((xinpsd_matrix[k], xPSD.reshape(-1,1)), axis = 1)
                yinpsd_matrix[k] = np.concatenate((yinpsd_matrix[k], yPSD.reshape(-1,1)), axis = 1)
                zinpsd_matrix[k] = np.concatenate((zinpsd_matrix[k], zPSD.reshape(-1,1)), axis = 1)


        counter += 1

    freqindex = find_nearest(xfreq,drivingfreq)

    for i in range(totalspheres):
        xinpsd_avg[i] = np.mean(xinpsd_matrix[i], axis=1)
        yinpsd_avg[i] = np.mean(yinpsd_matrix[i], axis=1)
        zinpsd_avg[i] = np.mean(zinpsd_matrix[i], axis=1)

        xg[i] = np.sqrt(xinpsd_avg[i][freqindex])
        yg[i] = np.sqrt(yinpsd_avg[i][freqindex])
        xg[i] = np.sqrt(zinpsd_avg[i][freqindex])

    return xinpsd_avg, yinpsd_avg, zinpsd_avg, xfreq, xg, yg, zg

sphere_folder = r'D:\Lab data\20250214\CH0 modulate'

no_sphere_folder = r'D:\Lab data\20250217\Amp modulation'

testfreqs, freqs_list = folder_sorting(sphere_folder)
#freqs_list = [int(f) for f in freqs_list]

color_map = mpl.colormaps.get_cmap('CMRmap')
colorind = np.linspace(0,0.8,len(freqs_list))
color_codes = [color_map(colorind[i]) for i in range(len(freqs_list))]
fig1,ax1 = plt.subplots(3,1)

plotfreqs = []
gains = np.empty((3,0))

for i in range(len(freqs_list)):
    
    xpsd_avg, ypsd_avg, zpsd_avg, xfreq, xg, yg, zg = file_frequency_space_TF(sphere_folder, testfreqs[freqs_list[i]], int(freqs_list[i]))
    print(xg)
    freq = xfreq[0,:]
    # figs[i], axs[i] = plt.subplots(3, 1, sharex=True, tight_layout=True)

    ax1[0].semilogy(freq, xpsd_avg,color=color_codes[i],label=freqs_list[i])
    ax1[1].semilogy(freq, ypsd_avg,color=color_codes[i],label=freqs_list[i])
    ax1[2].semilogy(freq, zpsd_avg,color=color_codes[i],label=freqs_list[i])



    plotfreqs.append(int(freqs_list[i]))
    spheregains = np.hstack((gains,np.vstack((xg,yg,zg)))) if gains.size else np.vstack((xg,yg,zg))
plt.show()
