"""
Created on Wed Sept 25 2024
Spring Constant Analysis for One Sphere in a Modulated Trap
"""

import os
import numpy as np
import scipy
import matplotlib as mpl 
import matplotlib.pyplot as plt
from scipy import fft
from sklearn.decomposition import PCA ## need this for principle component analysis below
from matplotlib.mlab import psd
from scipy.optimize import minimize, curve_fit
from scipy.stats import chi2, pearsonr
from scipy.signal import welch
from scipy.signal.windows import blackman
from scipy.signal import find_peaks, butter, lfilter, csd, coherence
from matplotlib.pyplot import gca
import h5py
import pandas as pd
from matplotlib.colors import LogNorm
import seaborn as sn
from scipy.interpolate import interp1d
from basichdf5 import *



def h5_position_file_folder_scan(main_directory, totalspheres):

    xasddata = [ [] for i in range(totalspheres)]
    yasddata = [ [] for i in range(totalspheres)]
    freqasddata = [ [] for i in range(totalspheres)]
    x_peak_scan = []
    y_peak_scan = []
    framerate_list = []

    os.chdir(main_directory)
    counter = 0
    for filename in sorted(os.listdir(main_directory)):
        if filename.endswith(".h5"):
            print(filename)
            xposdata, yposdata, xfftmatrix, yfftmatrix, frequency_bins, fs = hdf5_sphere_data_scraper(filename)

            framerate_list.append(fs)
    
            x_peaks_list = [[] for i in range(totalspheres)]
            y_peaks_list = [[] for i in range(totalspheres)]
            for i in range(totalspheres):

                x_peak_indices, x_peak_dict = find_peaks(xfftmatrix[:,i], height=3E-9)
                x_peak_heights = x_peak_dict['peak_heights']
                x_peak_freqs = frequency_bins[x_peak_indices]
                x_peaks = (np.vstack((x_peak_indices, x_peak_freqs, x_peak_heights))).T
                x_peaks_list[i] = x_peaks

                y_peak_indices, y_peak_dict = find_peaks(yfftmatrix[:,i], height=3E-9)
                y_peak_heights = y_peak_dict['peak_heights']
                y_peak_freqs = frequency_bins[y_peak_indices]
                y_peaks = (np.vstack((y_peak_indices, y_peak_freqs, y_peak_heights))).T
                y_peaks_list[i] = y_peaks
                
                
                if counter == 0:
                    freqasddata[i] = frequency_bins.reshape(-1,1)
                    xasddata[i] = xfftmatrix[:,i].reshape(-1,1)
                    yasddata[i] = yfftmatrix[:,i].reshape(-1,1)
                    
                    
                else:
                    freqasddata[i] = np.concatenate((freqasddata[i], frequency_bins.reshape(-1,1)), axis=1)
                    xasddata[i] = np.concatenate((xasddata[i], xfftmatrix[:,i].reshape(-1,1)), axis=1)
                    yasddata[i] = np.concatenate((yasddata[i], yfftmatrix[:,i].reshape(-1,1)), axis=1)
                    
            
            if x_peak_scan == []:
                x_peak_scan = [ x_peaks_list ]
                y_peak_scan = [ y_peaks_list ]
            
            else:
                x_peak_scan.append(x_peaks_list)
                y_peak_scan.append(y_peaks_list)
            
            counter +=1
                                   
    

    return x_peak_scan, y_peak_scan, freqasddata, xasddata, yasddata, framerate_list


def plot_responcevsfile(x_peak_scan, y_peak_scan, data_names, main_directory, totalspheres, savefigs):

    figb, axb = plt.subplots(1,2, tight_layout=True)
    #figb.set_size_inches(10,5)
    #figb.set_dpi(600)

    figc, axc = plt.subplots(1,2, tight_layout=True)
    #figc.set_size_inches(10,5)
    #figc.set_dpi(600)

    axb[0].set_title('X Peak vs Modulation')
    axb[1].set_title('Y Peak vs Modulation')
    axb[0].set_xlabel(r'Beam Position Modulation ($\mu m$)')
    axb[0].set_ylabel('Frequency Drift (Hz)')
    axb[1].set_xlabel(r'Beam Position Modulation ($\mu m$)')
    axb[1].set_ylabel('Frequency Drift (Hz)')

    axc[0].set_title('X Peak vs Modulation')
    axc[1].set_title('Y Peak vs Modulation')
    axc[0].set_xlabel(r'Beam Position Modulation ($\mu m$)')
    axc[0].set_ylabel(r'Amplitude ($m/ \sqrt{Hz}$)')
    axc[1].set_xlabel(r'Beam Position Modulation ($\mu m$)')
    axc[1].set_ylabel(r'Amplitude ($m/ \sqrt{Hz}$)')

    xreffreqs = []
    yreffreqs = []

    #get the frequencies of the last file in the folder for comparison
    for j in range(len(x_peak_scan[-1])):
        refpeaks = x_peak_scan[-1][j]
        refpeaksortindices = np.argsort(refpeaks[:,2])[::-1]
        for p in refpeaksortindices:
            
            if refpeaks[p,1] > 60:
                refmaxpeak = refpeaks[p,:]
                break
        xreffreqs.append(refmaxpeak[1])

    for j in range(len(y_peak_scan[-1])):
        refpeaks = y_peak_scan[-1][j]
        refpeaksortindices = np.argsort(refpeaks[:,2])[::-1]
        for p in refpeaksortindices:
            
            if refpeaks[p,1] > 60:
                refmaxpeak = refpeaks[p,:]
                break
        yreffreqs.append(refmaxpeak[1])



    for i in range(len(data_names)):

        for j in range(len(x_peak_scan[i])):
            xpeaks = x_peak_scan[i][j]
            peaksortindices = np.argsort(xpeaks[:,2])[::-1]
            for p in peaksortindices:
                
                if xpeaks[p,1] > 60:
                    maxpeak = xpeaks[p,:]
                    break
                
            
            normpeakfreq = maxpeak[1] - xreffreqs[j]
            axb[0].scatter(data_names[i], normpeakfreq, color = 'b', label =('Sphere ' + str(j)))
            
            axc[0].scatter(data_names[i], maxpeak[2], color = 'b', label =('Sphere ' + str(j)))
            
        
        for j in range(len(y_peak_scan[i])):
            ypeaks = y_peak_scan[i][j]
            peaksortindices = np.argsort(ypeaks[:,2])[::-1]
            for p in peaksortindices:
                
                if ypeaks[p,1] > 60:
                    maxpeak = ypeaks[p,:]
                    break

            
            normpeakfreq = maxpeak[1] - yreffreqs[j]
            axb[1].scatter(data_names[i], normpeakfreq, color = 'b', label =('Sphere ' + str(j)))
            
            axc[1].scatter(data_names[i], maxpeak[2], color = 'b', label =('Sphere ' + str(j)))
            

    handles, labels = axb[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    #axb[0].legend(by_label.values(), by_label.keys(), fontsize=12, loc="lower right", borderaxespad=1)

    handles, labels = axc[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    #axc[0].legend(by_label.values(), by_label.keys(), fontsize=12, borderaxespad=1)

    if savefigs:        
        figb.savefig(os.path.join(main_directory, 'freqshift.png'))   # save the figure to file 
        figc.savefig(os.path.join(main_directory, 'amplitudeshift.png'))   # save the figure to file

    plt.show()
    #plt.close('all')
    return

def plot_separation_ASD_scan(freqasddata, xasddata, yasddata, data_names, main_directory, savefigs, color_codes):

    figs={}
    axs={}
    for i in range(len(freqasddata)):
        figs[i], axs[i] = plt.subplots(2, 1, sharex=True, tight_layout=True)

        
        for j in range((freqasddata[i].shape)[1]):
            label_name = str(data_names[j]) + ' um'
            axs[i][0].semilogy(freqasddata[i][:,j], xasddata[i][:,j], color = color_codes[j], label=label_name)
            axs[i][1].semilogy(freqasddata[i][:,j], yasddata[i][:,j], color = color_codes[j], label=label_name)
        axs[i][0].set_xlim([5,350])
        axs[i][1].set_xlim([5,350])

        handles, labels = axs[i][0].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        axs[i][0].legend(by_label.values(), by_label.keys(), fontsize=12, loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0.1)
        #titlename = "Sphere " + str(i) + ' Response'
        #figs[i].suptitle(titlename)
        
        axs[i][1].set_xlabel('Frequency (Hz)')
        axs[i][0].set_ylabel(r'X ASD ($m/ \sqrt{Hz}$)')
        axs[i][1].set_ylabel(r'Y ASD ($m/ \sqrt{Hz}$)')
        
        if savefigs:
            figs[i].savefig(os.path.join(main_directory, titlename +'.png'))

    plt.show()
    #plt.close('all')
    return

main_directory = r"D:\Lab data\20240925"
totalspheres = 1
data_names = [0, 3.5, 5.25, 7]
saveflag = True
savefigs = False

color_map = mpl.colormaps.get_cmap('viridis')
colorind = np.linspace(0,0.8,len(data_names))
color_codes = [color_map(colorind[i]) for i in range(len(colorind))]

x_peak_scan, y_peak_scan, freqasddata, xasddata, yasddata, framerate_list = h5_position_file_folder_scan(main_directory, totalspheres)
plot_responcevsfile(x_peak_scan, y_peak_scan, data_names, main_directory, totalspheres, savefigs)
plot_separation_ASD_scan(freqasddata, xasddata, yasddata, data_names, main_directory, savefigs, color_codes)
