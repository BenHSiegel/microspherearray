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
        laser_data = hf.get('CH4')
        sum_data = hf.get('SUM')
        x_data = hf.get('X')
        y_data = hf.get('Y')
        sampleT = hf.attrs['Fsamp']

        laser_data = np.array(laser_data)
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
    xdata = np.array(x_data)
    ydata = np.array(y_data)
    zdata = np.array(sum_data)
    
    time = (np.arange(len(xdata[0,:]))) * sampleT
    hf.close()
    if fb_included == False:
        return xdata, ydata, zdata, laser_data, framerate, fb_included, None, None, None, None, None, None, None, None, None
    else:
        return xdata, ydata, zdata, None, framerate, fb_included, x_fb_data, y_fb_data, z_fb_data, xgains, ygains, zgains, xcorrection, ycorrection, zcorrection

def file_plotter(folder, filelist):
    '''
    Plots the PSD of the x, y, and z data from a list of files in a folder.
    folder = path to the folder containing the files
    filelist = list of files to plot
    '''
    counter = 0
    figs, axs = {}, {}
    figfall, axfall = {}, {}
    fallcounter = 0
    falllist = []
    for i in filelist:
        xdata, ydata, zdata, laser_data, framerate, fb_included, x_fb, y_fb, z_fb, xgains, ygains, zgains, xcorrection, ycorrection, zcorrection = hdf5_scraper(os.path.join(folder,i))
        if len(xdata.shape) > 1:
            totalspheres = xdata.shape[0]
            time = (np.arange(len(xdata[0,:]))) * (1/framerate)
        else:
            totalspheres = 1
            time = (np.arange(len(xdata))) * (1/framerate)

        # figs[counter], axs[counter] = plt.subplots(3, 1, sharex=True, tight_layout=True)
        # figs[counter].suptitle(i)
        # axs[counter][0].set_title('X motion')
        # axs[counter][1].set_title('Y motion')
        # axs[counter][2].set_title('Z motion')
        # axs[counter][0].set_ylabel('X position (m)')
        # axs[counter][1].set_ylabel('Y position (m)')
        # axs[counter][2].set_ylabel('Z position (m)')
        # axs[counter][2].set_xlabel('Time (s)')
        if totalspheres == 1:

            if min(zdata[0,:]) < 0.07 and 0 not in falllist: #sphere fell out since no backscatter
                falllist.append(0)
                figs[counter], axs[counter] = plt.subplots(4, 1, sharex=True, tight_layout=True)
                figs[counter].suptitle(i)
                axs[counter][0].set_title('X motion')
                axs[counter][1].set_title('Y motion')
                axs[counter][2].set_title('Z motion')
                axs[counter][3].set_title('Laser motion')
                axs[counter][0].set_ylabel('X position (V)')
                axs[counter][1].set_ylabel('Y position (V)')
                axs[counter][2].set_ylabel('Z position (V)')
                axs[counter][3].set_ylabel('Laser fluctuations (V)')
                axs[counter][3].set_xlabel('Time (s)')
                axs[counter][0].plot(time,xdata[0, :], label='Sphere 1')
                axs[counter][1].plot(time,ydata[0, :], label='Sphere 1')
                axs[counter][2].plot(time,zdata[0, :], label='Sphere 1')
                axs[counter][3].plot(time,laser_data[0, :], label='Sphere 1')
                segmentsize = framerate * 0.3 #scan in 0.3 second segments
                zderivative = np.diff(zdata[0, :])
                max_deriv_idx = np.argmax(np.abs(zderivative))
                figfall[fallcounter], axfall[fallcounter] = plt.subplots(4, 1, sharex=True, tight_layout=True)
                axfall[fallcounter][0].set_title(f'X motion before fall of sphere {0+1}')
                axfall[fallcounter][1].set_title(f'Y motion before fall of sphere {0+1}')
                axfall[fallcounter][2].set_title(f'Z motion before fall of sphere {0+1}')
                axfall[fallcounter][3].set_title(f'Laser motion before fall of sphere {0+1}')
                axfall[fallcounter][0].set_ylabel('X PSD (V^2/Hz)')
                axfall[fallcounter][1].set_ylabel('Y PSD (V^2/Hz)')
                axfall[fallcounter][2].set_ylabel('Z PSD (V^2/Hz)')
                axfall[fallcounter][3].set_ylabel('Laser PSD (V^2/Hz)')
                axfall[fallcounter][2].set_xlabel('Frequency (Hz)')
                axfall[fallcounter][0].set_xlim(5, 400)
                axfall[fallcounter][1].set_xlim(5, 400)
                axfall[fallcounter][2].set_xlim(5, 400)
                axfall[fallcounter][3].set_xlim(5, 400)
                figfall[fallcounter].suptitle(f'PSD before fall of sphere {0+1} in {i}')
                for k in range(0,10): #scan 10 segments before the fall
                    
                    start_idx = int(max_deriv_idx - k * segmentsize)
                    if start_idx < 0:
                        start_idx = 0
                    end_idx = int(start_idx + segmentsize)
                    xfreq, xPSD = welch(xdata[0, start_idx:end_idx], framerate, 'hann', segmentsize)
                    yfreq, yPSD = welch(ydata[0, start_idx:end_idx], framerate, 'hann', segmentsize)
                    zfreq, zPSD = welch(zdata[0, start_idx:end_idx], framerate, 'hann', segmentsize)
                    laser_freq, laser_PSD = welch(laser_data[0, start_idx:end_idx], framerate, 'hann', segmentsize)
                    fall_label = f'From {start_idx/framerate:.2f} to {(end_idx-1)/framerate:.2f} s'
                    axfall[fallcounter][0].semilogy(xfreq, xPSD, alpha=0.6, label=fall_label)
                    axfall[fallcounter][1].semilogy(yfreq, yPSD, alpha=0.6, label=fall_label)
                    axfall[fallcounter][2].semilogy(zfreq, zPSD, alpha=0.6, label=fall_label)
                    axfall[fallcounter][3].semilogy(laser_freq, laser_PSD, alpha=0.6, label=fall_label)

                    if start_idx == 0:
                        break
                axfall[fallcounter][0].legend(loc='upper left', bbox_to_anchor=(1.01, 1))
                fallcounter += 1


        else:
            for j in range(totalspheres):
                axs[counter][0].plot(time, xdata[j, :], label=f'Sphere {j+1}', alpha=0.6)
                axs[counter][1].plot(time, ydata[j, :], label=f'Sphere {j+1}', alpha=0.6)
                axs[counter][2].plot(time, zdata[j, :], label=f'Sphere {j+1}', alpha=0.6)

                if min(zdata[j,:]) < 0.08 and j not in falllist: #sphere fell out since no backscatter
                    falllist.append(j)
                    segmentsize = framerate * 0.2 #scan in 0.2 second segments
                    zderivative = np.diff(zdata[j, :])
                    max_deriv_idx = np.argmax(np.abs(zderivative))

                    for k in range(1,5): #scan 4 segments before the fall
                        figfall[fallcounter], axfall[fallcounter] = plt.subplots(3, 1, sharex=True, tight_layout=True)
                        start_idx = int(max_deriv_idx - k * segmentsize)
                        end_idx = int(start_idx + segmentsize)
                        xfreq, xPSD = welch(xdata[j, start_idx:end_idx], framerate, 'hann', segmentsize)
                        yfreq, yPSD = welch(ydata[j, start_idx:end_idx], framerate, 'hann', segmentsize)
                        zfreq, zPSD = welch(zdata[j, start_idx:end_idx], framerate, 'hann', segmentsize)
                        axfall[fallcounter][0].semilogy(xfreq, xPSD, alpha=0.6)
                        axfall[fallcounter][1].semilogy(yfreq, yPSD, alpha=0.6)
                        axfall[fallcounter][2].semilogy(zfreq, zPSD, alpha=0.6)
                        axfall[fallcounter][0].set_title(f'X motion before fall of sphere {j+1}')
                        axfall[fallcounter][1].set_title(f'Y motion before fall of sphere {j+1}')
                        axfall[fallcounter][2].set_title(f'Z motion before fall of sphere {j+1}')
                        axfall[fallcounter][0].set_ylabel('X PSD (V^2/Hz)')
                        axfall[fallcounter][1].set_ylabel('Y PSD (V^2/Hz)')
                        axfall[fallcounter][2].set_ylabel('Z PSD (V^2/Hz)')
                        axfall[fallcounter][2].set_xlabel('Frequency (Hz)')
                        axfall[fallcounter][0].set_xlim(10, 400)
                        axfall[fallcounter][1].set_xlim(10, 400)
                        axfall[fallcounter][2].set_xlim(10, 400)
                        figfall[fallcounter].suptitle(f'PSD before fall of sphere {j+1} in {i} from {start_idx/framerate:.2f} to {(end_idx-1)/framerate:.2f} seconds')
                        fallcounter += 1


            axs[counter][0].legend(loc='upper left', bbox_to_anchor=(1.01, 1))
        counter += 1
    plt.show()
    return

        


def file_psd_averager(folder, filelist):
    counter = 0
    #segmentsize = 2048
    for i in filelist:
        xdata, ydata, zdata, laser_data, framerate, fb_included, x_fb, y_fb, z_fb, xgains, ygains, zgains, xcorrection, ycorrection, zcorrection = hdf5_scraper(os.path.join(folder,i))
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

            if laser_data is not None:
                laserpsd_matrix = [[] for j in range(totalspheres)]
                laserpsd_avg = [[] for j in range(totalspheres)]

            
        for k in range(totalspheres):
            xfreq, xPSD = welch(xdata[k], framerate, 'hann', len(xdata[k]))
            yfreq, yPSD = welch(ydata[k], framerate, 'hann', len(ydata[k]))
            zfreq, zPSD = welch(zdata[k], framerate, 'hann', len(zdata[k]))

            if laser_data is not None:
                laser_freq, laser_PSD = welch(laser_data[k], framerate, 'hann', len(laser_data[k]))

            if counter == 0:
                xfreq = xfreq.reshape(1,-1)
                xpsd_matrix[k] = xPSD.reshape(-1,1)
                ypsd_matrix[k] = yPSD.reshape(-1,1)
                zpsd_matrix[k] = zPSD.reshape(-1,1)
                if laser_data is not None:
                    laserpsd_matrix[k] = laser_PSD.reshape(-1,1)
            else:
                xfreq = xfreq.reshape(1,-1)
                xpsd_matrix[k] = np.concatenate((xpsd_matrix[k], xPSD.reshape(-1,1)), axis = 1)
                ypsd_matrix[k] = np.concatenate((ypsd_matrix[k], yPSD.reshape(-1,1)), axis = 1)
                zpsd_matrix[k] = np.concatenate((zpsd_matrix[k], zPSD.reshape(-1,1)), axis = 1)
                if laser_data is not None:
                    laserpsd_matrix[k] = np.concatenate((laserpsd_matrix[k], laser_PSD.reshape(-1,1)), axis = 1)
        
        counter += 1
    
    for i in range(totalspheres):
        xpsd_avg[i] = np.mean(xpsd_matrix[i], axis=1)
        ypsd_avg[i] = np.mean(ypsd_matrix[i], axis=1)
        zpsd_avg[i] = np.mean(zpsd_matrix[i], axis=1)
        if laser_data is not None:
            laserpsd_avg[i] = np.mean(laserpsd_matrix[i], axis=1)
    
    if laser_data is None:
        laserpsd_avg = None

    return xpsd_avg, ypsd_avg, zpsd_avg, xfreq, laserpsd_avg


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


filepath = r'D:\Lab data\20250702'
groups, settings_list = folder_sorting(filepath)
print(settings_list)


# for i in range(len(settings_list)):
#     file_plotter(filepath, groups[settings_list[i]])


figs = {}
axs = {}
figx, axx = plt.subplots(1, 1, tight_layout=True)
figy, ayy = plt.subplots(1, 1, tight_layout=True)
figz, azz = plt.subplots(1, 1, tight_layout=True)
for i in range(len(settings_list)):
    xpsd, ypsd, zpsd, freq, laser_psd = file_psd_averager(filepath, groups[settings_list[i]])
    if laser_psd is not None:
        figs[i], axs[i] = plt.subplots(4, 1, sharex=True, tight_layout=True)
    else:
        figs[i], axs[i] = plt.subplots(3, 1, sharex=True, tight_layout=True)
    figs[i].suptitle(settings_list[i])
    axs[i][0].set_title('X motion')
    axs[i][1].set_title('Y motion')
    axs[i][2].set_title('Z motion')
    axs[i][0].set_ylabel('X PSD (V^2/Hz)')
    axs[i][1].set_ylabel('Y PSD (V^2/Hz)')
    axs[i][2].set_ylabel('Z PSD (V^2/Hz)')
    axs[i][0].semilogy(freq[0], xpsd[0])
    axs[i][1].semilogy(freq[0], ypsd[0])
    axs[i][2].semilogy(freq[0], zpsd[0])
    axs[i][0].set_xlim(1, 1000)
    axs[i][1].set_xlim(1, 1000)
    axs[i][2].set_xlim(1, 1000)
    if laser_psd is not None:
        axs[i][3].set_title('Laser fluctuations')
        axs[i][3].set_ylabel('Laser PSD (V^2/Hz)')
        axs[i][3].set_xlabel('Frequency (Hz)')
        axs[i][3].semilogy(freq[0], laser_psd[0])
        axs[i][3].set_xlim(1, 1000)

    else:
        axs[i][2].set_xlabel('Frequency (Hz)')

    label = (settings_list[i]).replace('_', ' ')
    axx.semilogy(freq[0], xpsd[0], label=label, alpha=0.6)
    ayy.semilogy(freq[0], ypsd[0], label=label, alpha=0.6)
    azz.semilogy(freq[0], zpsd[0], label=label, alpha=0.6)

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
axx.set_xlim(1, 1000)
ayy.set_xlim(1, 1000)
azz.set_xlim(1, 1000)
#axx.set_ylim(1e-10, 1e-3)
#ayy.set_ylim(1e-10, 1e-3)
#azz.set_ylim(1e-11, 1e-4)

plt.show()