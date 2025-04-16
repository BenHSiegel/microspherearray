'''
Scrapes hdf5 files to find a transfer function
'''

import numpy as np
import h5py
import scipy as sp
import matplotlib as mp
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
    hf.close()
    x = np.vstack((xin,xout))
    y = np.vstack((yin,yout))
    z = np.vstack((zin,zout))

    return xin, xout, yin, yout, zin, zout,Fs


def file_frequency_space_TF(folder, filelist, drivingfreq):
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

            xphi_matrix = [[] for j in range(totalspheres)]
            yphi_matrix = [[] for j in range(totalspheres)]
            zphi_matrix = [[] for j in range(totalspheres)]
            
            xphi_avg = [[] for j in range(totalspheres)]
            yphi_avg = [[] for j in range(totalspheres)]
            zphi_avg = [[] for j in range(totalspheres)]

            xg = [[] for j in range(totalspheres)]
            yg = [[] for j in range(totalspheres)]
            zg = [[] for j in range(totalspheres)]
            
        for k in range(totalspheres):
            xfreq, xPSD = welch(xin, framerate, 'hann', segmentsize)
            yfreq, yPSD = welch(yin, framerate, 'hann', segmentsize)
            zfreq, zPSD = welch(zin, framerate, 'hann', segmentsize)
            xfreqout, xPSDout = welch(xout, framerate, 'hann', segmentsize)
            yfreqout, yPSDout = welch(yout, framerate, 'hann', segmentsize)
            zfreqout, zPSDout = welch(zout, framerate, 'hann', segmentsize)

            xcorr = correlate(xin,xout)
            xlags = correlation_lags(len(xin), len(xout))
            xphi = xlags[np.argmax(xcorr)] * 2 * np.pi * drivingfreq / framerate

            ycorr = correlate(yin,yout)
            ylags = correlation_lags(len(yin), len(yout))
            yphi = ylags[np.argmax(ycorr)] * 2 * np.pi * drivingfreq / framerate

            zcorr = correlate(zin,zout)
            zlags = correlation_lags(len(zin), len(zout))
            zphi = zlags[np.argmax(zcorr)] * 2 * np.pi * drivingfreq / framerate

            if counter == 0:
                xfreq = xfreq.reshape(1,-1)
                xinpsd_matrix[k] = xPSD.reshape(-1,1)
                yinpsd_matrix[k] = yPSD.reshape(-1,1)
                zinpsd_matrix[k] = zPSD.reshape(-1,1)

                xoutpsd_matrix[k] = xPSDout.reshape(-1,1)
                youtpsd_matrix[k] = yPSDout.reshape(-1,1)
                zoutpsd_matrix[k] = zPSDout.reshape(-1,1)

                xphi_matrix[k] = xphi
                yphi_matrix[k] = yphi
                zphi_matrix[k] = zphi

            else:
                xfreq = xfreq.reshape(1,-1)
                xinpsd_matrix[k] = np.concatenate((xinpsd_matrix[k], xPSD.reshape(-1,1)), axis = 1)
                yinpsd_matrix[k] = np.concatenate((yinpsd_matrix[k], yPSD.reshape(-1,1)), axis = 1)
                zinpsd_matrix[k] = np.concatenate((zinpsd_matrix[k], zPSD.reshape(-1,1)), axis = 1)

                xoutpsd_matrix[k] = np.concatenate((xoutpsd_matrix[k], xPSDout.reshape(-1,1)), axis = 1)
                youtpsd_matrix[k] = np.concatenate((youtpsd_matrix[k], yPSDout.reshape(-1,1)), axis = 1)
                zoutpsd_matrix[k] = np.concatenate((zoutpsd_matrix[k], zPSDout.reshape(-1,1)), axis = 1)

                xphi_matrix.append(xphi)
                yphi_matrix.append(yphi)
                zphi_matrix.append(zphi)

        counter += 1
    
    for i in range(totalspheres):
        xinpsd_avg[i] = np.mean(xinpsd_matrix[i], axis=1)
        yinpsd_avg[i] = np.mean(yinpsd_matrix[i], axis=1)
        zinpsd_avg[i] = np.mean(zinpsd_matrix[i], axis=1)

        xoutpsd_avg[i] = np.mean(xoutpsd_matrix[i], axis=1)
        youtpsd_avg[i] = np.mean(youtpsd_matrix[i], axis=1)
        zoutpsd_avg[i] = np.mean(zoutpsd_matrix[i], axis=1)

        xphi_avg[i] = np.mean(xphi_matrix[i])
        yphi_avg[i] = np.mean(yphi_matrix[i])
        zphi_avg[i] = np.mean(zphi_matrix[i])

        xg[i] = np.sqrt(np.max(xoutpsd_avg[i])/(np.max(xinpsd_avg[i])))
        yg[i] = np.sqrt(np.max(youtpsd_avg[i])/(np.max(yinpsd_avg[i])))
        zg[i] = np.sqrt(np.max(zoutpsd_avg[i])/(np.max(zinpsd_avg[i])))
    
    return xinpsd_avg, yinpsd_avg, zinpsd_avg, xoutpsd_avg, youtpsd_avg, zoutpsd_avg, xfreq, xphi_avg, yphi_avg, zphi_avg, xg, yg, zg


#For finding filter TF
def hdf5_scraper_filtertype(filename):
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
    zfeedback= hf.get('Z Feedback')
    sampleT = hf.attrs['Fsamp']
    Fs = 1/sampleT
    xerror = np.array(xfeedback[0,:])
    xfilterin = np.array(xfeedback[1,:])
    xcoerce = np.array(xfeedback[2,:])
    xfilterout = np.array(xfeedback[3,:])
    zerror = np.array(zfeedback[0,:])
    zfilterin = np.array(zfeedback[1,:])
    zfilterout = np.array(zfeedback[2,:])
    zcompdelay = np.array(zfeedback[3,:])

    time = (np.arange(len(xerror))) * sampleT
    hf.close()

    return xerror, xcoerce, xfilterin, xfilterout, zerror, zcompdelay, zfilterin, zfilterout ,Fs


def file_filter_TF(folder, filelist, drivingfreq):
    counter = 0
    for i in filelist:
        xerror, xcoerce, xfilterin, xfilterout, zerror, zcompdelay, zfilterin, zfilterout, framerate = hdf5_scraper_filtertype(os.path.join(folder,i))

        totalspheres = 1
        if counter == 0:
            xinpsd_matrix = [[] for j in range(totalspheres)]
            zinpsd_matrix = [[] for j in range(totalspheres)]
            
            xinpsd_avg = [[] for j in range(totalspheres)]
            zinpsd_avg = [[] for j in range(totalspheres)]

            xoutpsd_matrix = [[] for j in range(totalspheres)]
            zoutpsd_matrix = [[] for j in range(totalspheres)]
            
            xoutpsd_avg = [[] for j in range(totalspheres)]
            zoutpsd_avg = [[] for j in range(totalspheres)]

            xphi_matrix = [[] for j in range(totalspheres)]
            zphi_matrix = [[] for j in range(totalspheres)]
            
            xphi_avg = [[] for j in range(totalspheres)]
            zphi_avg = [[] for j in range(totalspheres)]

            xg = [[] for j in range(totalspheres)]
            zg = [[] for j in range(totalspheres)]
            
        for k in range(totalspheres):
            xerrorfreq, xerrorPSD = welch(xerror, framerate, 'hann', len(xerror))
            xcoercefreq, xcoercePSD = welch(xcoerce, framerate, 'hann', len(xcoerce))
            xfilterinfreq, xfilterinPSD = welch(xfilterin, framerate, 'hann', len(xfilterin))
            xfilteroutfreq, xfilteroutPSD = welch(xfilterout, framerate, 'hann', len(xfilterout))

            zerrorfreq, zerrorPSD = welch(zerror, framerate, 'hann', len(zerror))
            zcompdelfreq, zcompdelPSD = welch(zcompdelay, framerate, 'hann', len(zcompdelay))
            zfilterinfreq, zfilterinPSD = welch(zfilterin, framerate, 'hann', len(zfilterin))
            zfilteroutfreq, zfilteroutPSD = welch(zfilterout, framerate, 'hann', len(zfilterout))

            xcorr = correlate(xfilterin,xfilterout)
            xlags = correlation_lags(len(xfilterin), len(xfilterout))
            xphi = xlags[np.argmax(xcorr)] * 2 * np.pi * drivingfreq / framerate


            zcorr = correlate(zfilterin,zfilterout)
            zlags = correlation_lags(len(zfilterin), len(zfilterout))
            zphi = zlags[np.argmax(zcorr)] * 2 * np.pi * drivingfreq / framerate

            if counter == 0:
                xfreq = xerrorfreq.reshape(1,-1)
                xinpsd_matrix[k] = xfilterinPSD.reshape(-1,1)
                zinpsd_matrix[k] = zfilterinPSD.reshape(-1,1)

                xoutpsd_matrix[k] = xfilteroutPSD.reshape(-1,1)
                zoutpsd_matrix[k] = zfilteroutPSD.reshape(-1,1)

                xphi_matrix[k] = xphi
                zphi_matrix[k] = zphi

            else:
                xfreq = xerrorfreq.reshape(1,-1)
                xinpsd_matrix[k] = np.concatenate((xinpsd_matrix[k], xfilterinPSD.reshape(-1,1)), axis = 1)
                zinpsd_matrix[k] = np.concatenate((zinpsd_matrix[k], zfilterinPSD.reshape(-1,1)), axis = 1)

                xoutpsd_matrix[k] = np.concatenate((xoutpsd_matrix[k], xfilteroutPSD.reshape(-1,1)), axis = 1)
                zoutpsd_matrix[k] = np.concatenate((zoutpsd_matrix[k], zfilteroutPSD.reshape(-1,1)), axis = 1)

                xphi_matrix.append(xphi)
                zphi_matrix.append(zphi)

        counter += 1
    
    freqindex = find_nearest(xfreq,drivingfreq)

    for i in range(totalspheres):
        xinpsd_avg[i] = np.mean(xinpsd_matrix[i], axis=1)
        zinpsd_avg[i] = np.mean(zinpsd_matrix[i], axis=1)

        xoutpsd_avg[i] = np.mean(xoutpsd_matrix[i], axis=1)
        zoutpsd_avg[i] = np.mean(zoutpsd_matrix[i], axis=1)

        xphi_avg[i] = np.mean(xphi_matrix[i])
        zphi_avg[i] = np.mean(zphi_matrix[i])

        xg[i] = np.sqrt(xoutpsd_avg[i][freqindex]/(xinpsd_avg[i][freqindex]))
        zg[i] = np.sqrt(zoutpsd_avg[i][freqindex]/(zinpsd_avg[i][freqindex]))
    
    return xinpsd_avg, zinpsd_avg, xoutpsd_avg, zoutpsd_avg, xfreq, xphi_avg, zphi_avg, xg, zg

def folder_sorting(directory):
    testfreqs = defaultdict(list)
    
    for filename in os.listdir(directory):
        basename, extension = os.path.splitext(filename)
        if extension == '.h5':
            frequency, _ = basename.split('Hz')
            testfreqs[frequency].append(filename)
            

    freqs_list = [i for i in testfreqs if testfreqs[i]!=testfreqs.default_factory()]
    return testfreqs, freqs_list

filepath = r'D:\Lab data\20250313\D test'

testfreqs, freqs_list = folder_sorting(filepath)
freqs = [int(f) for f in freqs_list]


figs = {}
axs = {}

plotfreqs = []
#use 2 for filter check and 3 for feedback check
phases = np.empty((3,0))
gains = np.empty((3,0))

for i in range(len(freqs_list)):
    
    #for feedback tf
    xinpsd_avg, yinpsd_avg, zinpsd_avg, xoutpsd_avg, youtpsd_avg, zoutpsd_avg, xfreq, xphi_avg, yphi_avg, zphi_avg, xg, yg, zg = file_frequency_space_TF(filepath, testfreqs[freqs_list[i]], int(freqs_list[i]))
    
    # freq = xfreq[0,:]
    # figs[i], axs[i] = plt.subplots(3, 1, sharex=True, tight_layout=True)

    # axs[i][0].semilogy(freq, xinpsd_avg[0])
    # axs[i][0].semilogy(freq, xoutpsd_avg[0])
    # axs[i][1].semilogy(freq, yinpsd_avg[0])
    # axs[i][1].semilogy(freq, youtpsd_avg[0])
    # axs[i][2].semilogy(freq, zinpsd_avg[0])
    # axs[i][2].semilogy(freq, zoutpsd_avg[0])
    # axs[i][0].set_ylabel('X PSD (V^2/Hz)')
    # axs[i][1].set_ylabel('Y PSD (V^2/Hz)')
    # axs[i][2].set_ylabel('Z PSD (V^2/Hz)')
    # axs[i][2].set_xlabel('Frequency (Hz)')
    # figs[i].suptitle(freqs_list[i])

    plotfreqs.append(int(freqs_list[i]))
    phases = np.hstack((phases,np.vstack((xphi_avg,yphi_avg,zphi_avg)))) if phases.size else np.vstack((xphi_avg,yphi_avg,zphi_avg))
    gains = np.hstack((gains,np.vstack((xg,yg,zg)))) if gains.size else np.vstack((xg,yg,zg))

    #for filter characterizing
    # xinpsd_avg, zinpsd_avg, xoutpsd_avg, zoutpsd_avg, xfreq, xphi_avg, zphi_avg, xg, zg = file_filter_TF(filepath, testfreqs[freqs_list[i]], int(freqs_list[i]))

    # plotfreqs.append(int(freqs_list[i]))
    # phases = np.hstack((phases,np.vstack((xphi_avg,zphi_avg)))) if phases.size else np.vstack((xphi_avg,zphi_avg))
    # gains = np.hstack((gains,np.vstack((xg,zg)))) if gains.size else np.vstack((xg,zg))

figa,axa = plt.subplots(2,1,sharex=True)
labels = ['X', 'Y', "Z"]
for i in range(3):
    axa[0].scatter(plotfreqs,gains[i,:]/(np.max(gains[i,:])), label=labels[i],s=(25-(i+1)**2))
    axa[1].scatter(plotfreqs,phases[i,:], label=labels[i],s=(30-10*i))

axa[0].set_xlim(0,550)
axa[1].set_xlim(0,550)
axa[0].set_ylim(0,0.3)
axa[1].set_ylim((-2*np.pi),2*np.pi)
#axa[0].set_ylim(0.6,1.1)
#axa[1].axhline(y=(0), color='r', linestyle='dotted',alpha=0.3,label='0')
axa[1].set_yticks(np.arange(-2*np.pi, 2*np.pi+0.01, np.pi/2))
#ticklabels = [r'$-\pi$',  r'$-3\pi/4$', r'$-\pi/2$', r'$-\pi/4$', '$0$', r'$\pi/4$', r'$\pi/2$',  r'$3\pi/4$',  r'$\pi$']
ticklabels = [r'$-2\pi$',r'$-3\pi/2$',r'$-\pi$', r'$-\pi/2$', '$0$',  r'$\pi/2$',  r'$\pi$',  r'$3\pi/2$', r'$2\pi$']
axa[1].set_yticklabels(ticklabels)
axa[0].legend(loc=2)
axa[1].legend(loc=1)
axa[0].set_ylabel('Gains')
axa[1].set_ylabel('Phase offset (rad)')
axa[1].set_xlabel('Frequency (Hz)')
figa.suptitle('Bode Plot for Derivative Action')
axa[0].grid(alpha=0.4)
axa[1].grid(alpha=0.4)
plt.show()