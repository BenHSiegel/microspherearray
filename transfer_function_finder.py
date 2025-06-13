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
import pandas as pd

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

    # Get the keys of the directories (groups) at the root level
    root_keys = list(hf.keys())
    if 'SUM' in root_keys:
        sum_data = hf.get('SUM')
        x_data = hf.get('X')
        y_data = hf.get('Y')
        sampleT = hf.attrs['Fsamp']
        fb_included = False

    elif 'Z Signal' in root_keys:
        sum_data = hf.get('Z Signal')
        x_data = hf.get('X Signal')
        y_data = hf.get('Y Signal')
        sampleT = hf.attrs['Fsamp']
        fb_included = True
        xfb = hf.get('X Feedback')
        yfb = hf.get('Y Feedback')
        zfb = hf.get('Z Feedback')
        xgains = xfb.attrs['Gains (P,I,D)']
        ygains = yfb.attrs['Gains (P,I,D)']
        zgains = zfb.attrs['Gains (P,I,D)']
        # Extract feedback data for each axis and position
        x_fb_data = {
            'In': np.array(xfb.get('In')),
            'Middle': np.array(xfb.get('Middle')),
            'Out': np.array(xfb.get('Out'))
        }
        y_fb_data = {
            'In': np.array(yfb.get('In')),
            'Middle': np.array(yfb.get('Middle')),
            'Out': np.array(yfb.get('Out'))
        }
        z_fb_data = {
            'In': np.array(zfb.get('In')),
            'Middle': np.array(zfb.get('Middle')),
            'Out': np.array(zfb.get('Out'))
        }

        

        if xfb.attrs['Setpoint correction?'] == 'True':
            xcorrection = True
            ycorrection = True
        else:
            xcorrection = False
            ycorrection = False
        if zfb.attrs['Setpoint correction?'] == 'True':
            zcorrection = True
        else:
            zcorrection = False


    else:
        raise ValueError("The file does not contain the expected groups.")
    
    framerate = 1/sampleT
    xdata = np.array(x_data[0,:])
    ydata = np.array(y_data[0,:])
    zdata = np.array(sum_data[0,:])
    time = (np.arange(len(xdata))) * sampleT
    hf.close()
    if fb_included == False:
        return xdata, ydata, zdata, framerate, fb_included, None, None, None, None, None, None, None, None, None
    else:
        return xdata, ydata, zdata, framerate, fb_included, x_fb_data, y_fb_data, z_fb_data, xgains, ygains, zgains, xcorrection, ycorrection, zcorrection


def file_psd_averager(folder, filelist):
    counter = 0
    #segmentsize = 2048
    for i in filelist:
        xdata, ydata, zdata, framerate, fb_included, x_fb, y_fb, z_fb, xgains, ygains, zgains, xcorrection, ycorrection, zcorrection = hdf5_scraper(os.path.join(folder,i))
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
            xfreq, xPSD = welch(xdata, framerate, 'hann', len(xdata))
            yfreq, yPSD = welch(ydata, framerate, 'hann', len(ydata))
            zfreq, zPSD = welch(zdata, framerate, 'hann', len(zdata))
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
            if '_beamsorted_' in basename:
                settings, session = basename.split('_beamsorted_')
                groups[settings].append(filename)
            elif '_transferfunction_' in basename:
                settings, session = basename.split('_transferfunction_')
                groups[settings].append(filename)
            else:
                continue
            
            

    settings_list = [i for i in groups if groups[i]!=groups.default_factory()]
    
    return groups, settings_list


filepath = r'D:\Lab data\20250603\sphere 1'
#xdata, ydata, zdata, framerate, fb_included, x_fb, y_fb, z_fb, xgains, ygains, zgains, xcorrection, ycorrection, zcorrection = hdf5_scraper(os.path.join(filepath,filename))


groups, settings_list = folder_sorting(filepath)

print(settings_list)
figs = {}
axs = {}
figx, axx = plt.subplots(1, 1, tight_layout=True)
figy, ayy = plt.subplots(1, 1, tight_layout=True)
figz, azz = plt.subplots(1, 1, tight_layout=True)
for i in range(len(settings_list)):
    xpsd, ypsd, zpsd, freq = file_psd_averager(filepath, groups[settings_list[i]])
    
    if settings_list[i][0] == 'n':
        label = (settings_list[i]).replace('_', ' ')
        axx.semilogy(freq[0], xpsd[0], label=label)
        ayy.semilogy(freq[0], ypsd[0], label=label)
        azz.semilogy(freq[0], zpsd[0], label=label)

    if settings_list[i][0] == 'X':
        axx.semilogy(freq[0], xpsd[0], label=('D=' + settings_list[i][4:]).replace('_', '.'))
    elif settings_list[i][0] == 'Y':
        ayy.semilogy(freq[0], ypsd[0], label=('D=' + settings_list[i][4:]).replace('_', '.'))
    elif settings_list[i][0] == 'Z':
        label=(settings_list[i]).replace('_', ' ')
        azz.semilogy(freq[0], zpsd[0], label=label)

    # figs[i], axs[i] = plt.subplots(3, 1, sharex=True, tight_layout=True)

    # axs[i][0].semilogy(freq[0], xpsd[0])
    # axs[i][1].semilogy(freq[0], ypsd[0])
    # axs[i][2].semilogy(freq[0], zpsd[0])
    # axs[i][0].set_ylabel('X PSD (V^2/Hz)')
    # axs[i][1].set_ylabel('Y PSD (V^2/Hz)')
    # axs[i][2].set_ylabel('Z PSD (V^2/Hz)')
    # axs[i][2].set_xlabel('Frequency (Hz)')
    # figs[i].suptitle(settings_list[i])


axx.set_ylabel('X PSD (V^2/Hz)')
axx.set_xlabel('Frequency (Hz)')
ayy.set_ylabel('Y PSD (V^2/Hz)')
ayy.set_xlabel('Frequency (Hz)')
azz.set_ylabel('Z PSD (V^2/Hz)')
azz.set_xlabel('Frequency (Hz)')
axx.legend()
ayy.legend()
azz.legend()
axx.set_title('X PSD')
ayy.set_title('Y PSD')
azz.set_title('Z PSD')
axx.set_xlim(0, 1000)
ayy.set_xlim(0, 1000)
azz.set_xlim(0, 1000)
axx.set_ylim(1e-10, 1e-3)
ayy.set_ylim(1e-10, 1e-3)
azz.set_ylim(1e-11, 1e-4)

plt.show()