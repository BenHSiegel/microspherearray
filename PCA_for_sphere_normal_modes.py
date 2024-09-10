import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.signal import find_peaks
from matplotlib.pyplot import gca
import h5py
import os

from sklearn import decomposition


def filepositionreader(filename):
    '''
    Reads a HDF5 file and saves the position data from it for all the spheres
    as a column stack array
    
    Inputs
    -------
    filename:       Name of the hdf5 file to read from

    Returns
    -------
    pos_stack_x:    The x position data for each sphere (x1, x2, x3, ...)
    pos_stack_y:    The y position data for each sphere (y1, y2, y3, ...)
    pos_stack:      A stack of all the position data (x1, y1, x2, y2, ...)
    '''
    hf = h5py.File(filename, 'r')
    group = hf.get('position')
    framerate = group.attrs['framerate (fps)']
    
    pos_stack_x = []
    pos_stack_y = []
    pos_stack = []
    
    counter = 0
    for i in group.items():
        pos_i = np.array(i[1])
        
        if counter == 0:
            pos_stack_x = pos_i[:,1].reshape(-1,1)
            pos_stack_y = pos_i[:,2].reshape(-1,1)
            pos_stack = pos_i[:,1].reshape(-1,1)
            pos_stack = np.concatenate((pos_stack, pos_i[:,2].reshape(-1,1)), axis=1)
        else:
            pos_stack_x = np.concatenate((pos_stack_x, pos_i[:,1].reshape(-1,1)), axis=1)
            pos_stack_y = np.concatenate((pos_stack_y, pos_i[:,2].reshape(-1,1)), axis=1)
            pos_stack = np.concatenate((pos_stack, pos_i[:,1].reshape(-1,1), pos_i[:,2].reshape(-1,1)), axis=1)
            
        counter += 1
    
    hf.close()
    return pos_stack_x, pos_stack_y, pos_stack, framerate


def pcafitting(data, framerate):
    '''
    
    Parameters
    ----------
    data :      array of position data
    framerate : (int) camera framerate for the videos

    Returns
    -------
    freq:       Vector of the fft frequency bins
    PSDlist:    List of vectors of the PSDs from the pca transformed data

    '''
    segmentsize = round(framerate/4)
    fftbinning = 2048
    
    pca = decomposition.PCA() #'mle'
    pca.fit(data) ## fit our data
    orig = pca.transform(data) ## reconstruct the original data from the PCA transform
    print(len(orig))
    PSDlist = [[] for i in range(orig.shape[1])]
    
    for i in range(orig.shape[1]):
    
        freq, dataPSD_uncor = welch(orig[:,i], framerate, 'hann', segmentsize, segmentsize/4, fftbinning, 'constant', True, 'density', 0,'mean')
        
        PSDlist[i] = dataPSD_uncor
    
    return freq, PSDlist, orig
    

def normalmodedecomp(path):
    '''
    Takes all the time streams recorded from the videos
    Performs a PCA transform on them and then generates the ASD for each video
    At the end it RMS averages the ASDs and outputs plots
    
    Inputs
    -------
    

    Returns
    -------

    '''
    
    hdf5_list = []
    for filename in sorted(os.listdir(path)):
        if filename.endswith(".h5"):
            if filename[-4] != 'g':
                hdf5_list.append(filename)
                
                
    mpl.rcParams['figure.dpi'] = 600
    figs = {}
    axs = {}
    for i in hdf5_list:
        pos_stack_x, pos_stack_y, pos_stack, framerate = filepositionreader(i)

        freqx, xPSDs, origx = pcafitting(pos_stack_x, framerate)
        freqy, yPSDs, origy = pcafitting(pos_stack_y, framerate)
        freq, totPSDs, origtot = pcafitting(pos_stack, framerate) 
        
        figs[i], axs[i] = plt.subplots(3, 1, sharex=False)
        figs[i].set_size_inches(10.5, 18.5)
        
        for j in range(len(totPSDs)):
            axs[i][0].semilogy(freq, totPSDs[j])
        
        for j in range(len(yPSDs)):
            axs[i][1].semilogy(freqy, yPSDs[j])
        
        for j in range(len(xPSDs)):
            axs[i][2].semilogy(freqx, xPSDs[j])
           
        axs[i][2].set_title('X coordinate PCA Attempt')
        axs[i][2].set_xlim(0,150)
        axs[i][2].set_xlabel('Frequency [HZ]')
        axs[i][2].set_ylabel('PSD [arb.]')

        axs[i][1].set_title('Y coordinate PCA Attempt')
        axs[i][1].set_xlim(0,150)
        axs[i][1].set_xlabel('Frequency [HZ]')
        axs[i][1].set_ylabel('PSD [arb.]')

        axs[i][0].set_title('All coordinate PCA Attempt')
        axs[i][0].set_xlim(0,150)
        axs[i][0].set_xlabel('Frequency [HZ]')
        axs[i][0].set_ylabel('PSD [arb.]')

    return origtot, totPSDs

###############################################################################
  
# path = r"C:\Users\bensi\Documents\Research\20240319\middle"
# os.chdir(path)
# orig, psds = normalmodedecomp(path)