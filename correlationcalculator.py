# -*- coding: utf-8 -*-
"""
Created on Thu May 16 11:42:25 2024
Correlation matrix generator for spheres
@author: yalem
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
from scipy.stats import chi2
from scipy.signal import welch
from scipy.signal.windows import blackman
from scipy.signal import find_peaks, butter, lfilter
from matplotlib.pyplot import gca
import h5py
import pandas as pd
from matplotlib.colors import LogNorm

main_directory = r"C:\Users\Ben\Documents\Research\20240604"
totalspheres = 2
saveflag = True


def butter_highpass(data, highpassfq, fs, order=3):
    nyq = 0.5 * fs
    cornerfq = highpassfq / nyq
    b, a = butter(order, cornerfq, btype='highpass')
    filtered_data = lfilter(b, a, data)
    return filtered_data
    
def hdf5file_correlationprocessing(path, totalspheres, sep, saveflag, savename):
    hdf5_list = []
    for filename in sorted(os.listdir(path)):
        if filename.endswith(".h5") and not filename.endswith("avg.h5") and not filename.endswith("matrix.h5") :
            hdf5_list.append(filename)
    
    xcorrlist = []
    ycorrlist = []
    
    counter = 0
    for i in hdf5_list:
        hf = h5py.File(i, 'r')
        group = hf.get('position')
        fs = group.attrs['framerate (fps)']
    
        xposdata = []
        yposdata = []
        
        l = 0
        for j in group.items():
            pos = np.array(j[1])
            xpos = pos[:,1].reshape(-1,1)
            xfiltered = butter_highpass(xpos, 40, fs)
            ypos = pos[:,2].reshape(-1,1)
            yfiltered = butter_highpass(ypos, 40, fs)
            if l == 0:
                xposdata = xfiltered[:,0].reshape(-1,1)
                yposdata = yfiltered[:,0].reshape(-1,1)
            else:
                xposdata = np.concatenate((xposdata, xfiltered[:,0].reshape(-1,1)), axis=1)
                yposdata = np.concatenate((yposdata, yfiltered[:,0].reshape(-1,1)), axis=1)
                
            l+=1
            
        hf.close()
        
        xdf = pd.DataFrame(xposdata)
        ydf = pd.DataFrame(yposdata)
        
        xcorrmatrix = xdf.corr()
        ycorrmatrix = ydf.corr()    
        
        if counter == 0:
            xcorrlist = xcorrmatrix
            ycorrlist = ycorrmatrix
        else:
            xcorrlist = np.dstack((xcorrlist,xcorrmatrix))
            ycorrlist = np.dstack((ycorrlist,ycorrmatrix))
        counter += 1
            
    xcorr_averaged = np.mean(xcorrlist, axis=2)
    ycorr_averaged = np.mean(ycorrlist, axis=2)
    
    if saveflag:
        savename = savename + '.h5'
        hf = h5py.File(savename, 'w')
        hf.attrs.create('framerate (fps)', fs)
        hf.attrs.create('Separation (MHz)', sep)
        hf.attrs.create('Number of spheres', totalspheres)
        hf.create_dataset('X_correlations', data=xcorr_averaged)
        hf.create_dataset('Y_correlations', data=ycorr_averaged)

        hf.close()
    
    return xcorr_averaged, ycorr_averaged


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current Axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """
    
    norm = mpl.colors.LogNorm()
    
    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(np.absolute(data), norm=norm, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.6, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = mpl.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(np.absolute(data[i, j])) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts




separation_scan = []
x_peak_scan = []
y_peak_scan = []
correlation_scan = []

for path, folders, files in os.walk(main_directory):
    
    for folder_name in folders:
        directory = f"{path}/{folder_name}"
        os.chdir(directory)
        print(folder_name)
        with open('info.txt') as file:
            lines = [line.rstrip() for line in file]
        framerate = float(lines[0])
        sep = float(lines[1])
        data_label = lines[2]
        include_in_scan = lines[3]

        savename = str(sep) + 'correlationmatrix'
        xcorr_averaged, ycorr_averaged = hdf5file_correlationprocessing(directory, totalspheres, sep, saveflag, savename)

        if include_in_scan == 'True':
            xcorr_offdiags = []
            ycorr_offdiags = []
            for i in range(np.shape(xcorr_averaged)[0]):
                for j in range(np.shape(xcorr_averaged)[1]):
                    if j > i:
                        xcorr_offdiags.append(xcorr_averaged[i][j])
                        ycorr_offdiags.append(xcorr_averaged[i][j])
        
            correlation_scan.append((np.vstack((xcorr_offdiags, ycorr_offdiags))).T)

        for filename in sorted(os.listdir(directory)):
            if filename.endswith("rmsavg.h5"):
                hfpsd = h5py.File(filename, 'r')
                freqset = hfpsd.get('frequencies')
                xpsdset = hfpsd.get('XASD RMS Avg')
                ypsdset = hfpsd.get('YASD RMS Avg')
                freq = freqset[()]
                xpsd = xpsdset[()]
                ypsd = ypsdset[()]

                hfpsd.close()
        
        
        umsep = sep * 70
        fig, ax = plt.subplots(2, 2, gridspec_kw={'width_ratios': [1, 3]})
        fig.tight_layout()
        fig.set_size_inches(11, 8.5)
        fig.set_dpi(600)
        fig.suptitle(data_label, fontsize=18)
        plt.subplots_adjust(top=0.9)
        
        spherenames = [str(x) for x in range(totalspheres)]
        im, cbar = heatmap(xcorr_averaged, spherenames, spherenames, ax=ax[0,0],
                   cmap="YlGn")
        texts = annotate_heatmap(im, data=xcorr_averaged, valfmt="{x:.3f}")
        ax[0,0].set_title("X Correlation")
        im, cbar = heatmap(ycorr_averaged, spherenames, spherenames, ax=ax[1,0],
                   cmap="YlGn")
        texts = annotate_heatmap(im, data=ycorr_averaged, valfmt="{x:.3f}")
        ax[1,0].set_title("Y Correlation")

        Legend = []
        x_peaks_list = [[] for i in range(totalspheres)]
        y_peaks_list = [[] for i in range(totalspheres)]
        for i in range(totalspheres):
            ax[0,1].semilogy(freq, xpsd[i,:], linewidth=2)
            ax[1,1].semilogy(freq, ypsd[i,:], linewidth=2)
            Legend.append('Sphere ' + str(i))

            if include_in_scan == 'True':
                x_peak_indices, x_peak_dict = find_peaks(xpsd[i,:], height=3E-9)
                x_peak_heights = x_peak_dict['peak_heights']
                x_peak_freqs = freq[x_peak_indices]
                x_peaks = (np.vstack((x_peak_indices, x_peak_freqs, x_peak_heights))).T
                x_peaks_list[i] = x_peaks

                y_peak_indices, y_peak_dict = find_peaks(ypsd[i,:], height=3E-9)
                y_peak_heights = y_peak_dict['peak_heights']
                y_peak_freqs = freq[y_peak_indices]
                y_peaks = (np.vstack((y_peak_indices, y_peak_freqs, y_peak_heights))).T
                y_peaks_list[i] = y_peaks
        
        if include_in_scan == 'True':
            if x_peak_scan == []:
                x_peak_scan = [ x_peaks_list ]
                y_peak_scan = [ y_peaks_list ]
            
            else:
                x_peak_scan.append(x_peaks_list)
                y_peak_scan.append(y_peaks_list)
            separation_scan.append(umsep)
                        
        ax[0,1].grid()
        ax[0,1].set_xlim(2,350)
        ax[0,1].set_xlabel('Frequency (Hz)')
        ax[0,1].set_ylabel(r'ASD ($m/ \sqrt{Hz}$)')
        #ax[0,1].legend(Legend, fontsize=12, bbox_to_anchor=(1.04, 0), loc="lower left", borderaxespad=0)
        ax[0,1].set_title('X motion RMS Avg ASD')
        
        ax[1,1].grid()
        ax[1,1].set_xlim(2,350)
        ax[1,1].set_xlabel('Frequency (Hz)')
        ax[1,1].set_ylabel(r'ASD ($m/ \sqrt{Hz}$)')
        ax[1,1].legend(Legend, loc="upper right", borderaxespad=1.5)
        ax[1,1].set_title('Y motion RMS Avg ASD')
        
        fig.savefig(os.path.join(main_directory, data_label + '.png'))   # save the figure to file
        
    break

figa, axa = plt.subplots(1,2)
figa.tight_layout()
figa.set_size_inches(10,5)
figa.set_dpi(600)

figb, axb = plt.subplots(1,2)
figb.tight_layout()
figb.set_size_inches(10,5)
figb.set_dpi(600)

figc, axc = plt.subplots(1,2)
figc.tight_layout()
figc.set_size_inches(10,5)
figc.set_dpi(600)

color_codes = ['#FF8C00', '#000080', '#008000', '#00FF00', '#0000FF', '#00FFFF', '#FF0000', '#FF00FF', '#800000', '#808000', '#800080', '#008080']


axa[0].set_title('X Correlation vs Separation')
axa[1].set_title('Y Correlation vs Separation')
axa[0].set_xlabel(r'Separation ($\mu m$)')
axa[0].set_ylabel('Correlation')
axa[1].set_xlabel(r'Separation ($\mu m$)')
axa[1].set_ylabel('Correlation')

axb[0].set_title('X Peak vs Separation')
axb[1].set_title('Y Peak vs Separation')
axb[0].set_xlabel(r'Separation ($\mu m$)')
axb[0].set_ylabel('Frequency Drift (Hz)')
axb[1].set_xlabel(r'Separation ($\mu m$)')
axb[1].set_ylabel('Frequency Drift (Hz)')

axc[0].set_title('X Peak vs Separation')
axc[1].set_title('Y Peak vs Separation')
axc[0].set_xlabel(r'Separation ($\mu m$)')
axc[0].set_ylabel(r'Amplitude ($m/ \sqrt{Hz}$)')
axc[1].set_xlabel(r'Separation ($\mu m$)')
axc[1].set_ylabel(r'Amplitude ($m/ \sqrt{Hz}$)')



cor_legend = []
for i in range(totalspheres):
    for j in range(totalspheres):
        if j > i:
            cor_legend.append(str(i) + '-' + str(j))
xreffreqs = []
yreffreqs = []

#get the frequencies of them when furthest apart in the scan for comparison
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




for i in range(len(separation_scan)):
    for j in range(len(correlation_scan[i][:,0])):
        axa[0].scatter(separation_scan[i], correlation_scan[i][j,0], color=color_codes[j], label=cor_legend[j])
        axa[1].scatter(separation_scan[i], correlation_scan[i][j,1], color=color_codes[j], label=cor_legend[j])
    
    
    for j in range(len(x_peak_scan[i])):
        xpeaks = x_peak_scan[i][j]
        peaksortindices = np.argsort(xpeaks[:,2])[::-1]
        for p in peaksortindices:
            
            if xpeaks[p,1] > 60:
                maxpeak = xpeaks[p,:]
                break
            
        
        normpeakfreq = maxpeak[1] - xreffreqs[j]
        axb[0].scatter(separation_scan[i], normpeakfreq, color=color_codes[j], label =('Sphere ' + str(j)))
        
        axc[0].scatter(separation_scan[i], maxpeak[2], color=color_codes[j], label =('Sphere ' + str(j)))
        
    
    for j in range(len(y_peak_scan[i])):
        ypeaks = y_peak_scan[i][j]
        peaksortindices = np.argsort(ypeaks[:,2])[::-1]
        for p in peaksortindices:
            
            if ypeaks[p,1] > 60:
                maxpeak = ypeaks[p,:]
                break

        
        normpeakfreq = maxpeak[1] - yreffreqs[j]
        axb[1].scatter(separation_scan[i], normpeakfreq, color=color_codes[j], label =('Sphere ' + str(j)))
        
        axc[1].scatter(separation_scan[i], maxpeak[2], color=color_codes[j], label =('Sphere ' + str(j)))
        



handles, labels = axa[0].get_legend_handles_labels()
by_label = dict(zip(labels, handles))
axa[0].legend(by_label.values(), by_label.keys(), fontsize=12, loc="lower right", borderaxespad=1)



handles, labels = axb[0].get_legend_handles_labels()
by_label = dict(zip(labels, handles))
axb[0].legend(by_label.values(), by_label.keys(), fontsize=12, borderaxespad=1)

handles, labels = axc[0].get_legend_handles_labels()
by_label = dict(zip(labels, handles))
axc[0].legend(by_label.values(), by_label.keys(), fontsize=12, borderaxespad=1)

figa.savefig(os.path.join(main_directory, 'correlation.png'))   # save the figure to file

figb.savefig(os.path.join(main_directory, 'freqshift.png'))   # save the figure to file

figc.savefig(os.path.join(main_directory, 'amplitudeshift.png'))   # save the figure to file


