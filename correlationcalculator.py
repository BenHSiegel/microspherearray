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
            xfiltered = butter_highpass(xpos, 75, fs)
            ypos = pos[:,2].reshape(-1,1)
            yfiltered = butter_highpass(ypos, 75, fs)
            if l == 0:
                xposdata = xfiltered[:,0].reshape(-1,1)
                yposdata = yfiltered[:,0].reshape(-1,1)
            else:
                xposdata = np.concatenate((xposdata, xfiltered[:,0].reshape(-1,1)), axis=1)
                yposdata = np.concatenate((yposdata, yfiltered[:,0].reshape(-1,1)), axis=1)
                
            l+=1
            
        hf.close()
        
        xdf = pd.DataFrame(xposdata)  #error
        ydf = pd.DataFrame(yposdata)  #error
        
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


main_directory = r"D:\Lab data\20240513\part 2"
totalspheres = 3
saveflag = False

for path, folders, files in os.walk(main_directory):
    for folder_name in folders:
        directory = f"{path}/{folder_name}"
        os.chdir(directory)
        print(folder_name)
        with open('info.txt') as file:
            lines = [line.rstrip() for line in file]
        framerate = float(lines[0])
        sep = float(lines[1])

        savename = str(sep) + 'correlationmatrix'
        xcorr_averaged, ycorr_averaged = hdf5file_correlationprocessing(directory, totalspheres, sep, saveflag, savename)
    
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
        fig.suptitle(str(umsep) + ' um Separation Between Spheres', fontsize=18)
        plt.subplots_adjust(top=0.9)
        
        spherenames = ['1', '2', '3']
        im, cbar = heatmap(xcorr_averaged, spherenames, spherenames, ax=ax[0,0],
                   cmap="YlGn")
        texts = annotate_heatmap(im, data=xcorr_averaged, valfmt="{x:.3f}")
        ax[0,0].set_title("X Correlation")
        im, cbar = heatmap(ycorr_averaged, spherenames, spherenames, ax=ax[1,0],
                   cmap="YlGn")
        texts = annotate_heatmap(im, data=ycorr_averaged, valfmt="{x:.3f}")
        ax[1,0].set_title("Y Correlation")

        Legend = []
        
        for i in range(totalspheres):
            ax[0,1].semilogy(freq, xpsd[i,:], linewidth=2)
            ax[1,1].semilogy(freq, ypsd[i,:], linewidth=2)
            Legend.append('Sphere ' + str(i))
                        
        ax[0,1].grid()
        #axc.set_xlim(5,180)
        ax[0,1].set_xlabel('Frequency [Hz]')
        ax[0,1].set_ylabel(r'ASD [$m/ \sqrt{Hz}$]')
        #ax[0,1].legend(Legend, fontsize=12, bbox_to_anchor=(1.04, 0), loc="lower left", borderaxespad=0)
        ax[0,1].set_title('X motion RMS Avg ASD')
        
        ax[1,1].grid()
        #axc.set_xlim(5,180)
        ax[1,1].set_xlabel('Frequency [Hz]')
        ax[1,1].set_ylabel(r'ASD [$m/ \sqrt{Hz}$]')
        ax[1,1].legend(Legend, loc="upper right", borderaxespad=1.5)
        ax[1,1].set_title('Y motion RMS Avg ASD')
        
    break