# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 11:43:57 2024

@author: Ben
"""
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from matplotlib.collections import PolyCollection

from scipy.signal import find_peaks
from matplotlib.pyplot import gca
import h5py

def polygon_under_graph(x, y,base):
    """
    Construct the vertex list which defines the polygon filling the space under
    the (x, y) line graph. This assumes x is in ascending order.
    """
    return [(x[0], base), *zip(x, y), (x[-1], base)]

def multisphere3d(path, color_codes):
    hf = h5py.File(path, 'r')
    freqs = np.array(hf.get('frequencies'))
    X_asd = np.array(hf.get('XASD RMS Avg'))
    Y_asd = np.array(hf.get('YASD RMS Avg'))
    
    index = np.arange(1,26,1)
    g = np.ones(np.size(X_asd,1))

    figx = plt.figure()
    ax = figx.add_subplot(projection="3d")
    figy = pl.figure()
    ay = figy.add_subplot(projection="3d")
    verts = [polygon_under_graph(freqs[123:460],np.log10(X_asd[i,123:460]**2),-17.) for i in range(len(X_asd))]
    poly= PolyCollection(verts, facecolors=color_codes, alpha=0.3)
    lines = [ax.plot3D(freqs[123:460],g[123:460]*(i+1),np.log10(X_asd[i,123:460]**2), color=color_codes[i], alpha = 0.8, linewidth = 3) for i in range(len(X_asd))]
    ax.add_collection3d(poly,zs=index,zdir='y')
    ax.set(xlim=(60, 225), ylim=(0, 26), zlim=(-17,-14))
    ax.set_xlabel('Frequency (Hz)', fontsize=14, labelpad=10)
    ax.set_ylabel('Sphere', fontsize=14, labelpad=10)
    
    ax.grid(False)
    ax.tick_params(labelsize=12)
    #ax.tick_params(axis='both', top=False, labeltop = False, bottom = True, labelbottom = True, left = True, right = False, labelleft=True, labelright=False)
    ax.xaxis.pane.fill = False # Left pane
    ax.yaxis.pane.fill = False # Right pane
    ax.zaxis.pane.fill = False # bottom pane
    ax.set_zticks([])
    # # Transparent spines
    # ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    # ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    # # Transparent panes
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    verts = [polygon_under_graph(freqs[82:310],np.log10(Y_asd[i,82:310]**2),-17.) for i in range(len(X_asd))]
    poly= PolyCollection(verts, facecolors=color_codes, alpha=0.3)
    lines = [ay.plot3D(freqs[82:310],g[82:310]*(i+1),np.log10(Y_asd[i,82:310]**2), color=color_codes[i], alpha = 0.8, linewidth = 3) for i in range(len(Y_asd))]
    ay.add_collection3d(poly,zs=index,zdir='y')
    ay.set(xlim=(40, 150), ylim=(0, 26), zlim=(-17,-14.5))
    ay.set_xlabel('Frequency (Hz)', fontsize=14, labelpad=10)
    ay.set_ylabel('Sphere', fontsize=14, labelpad=10)
    ay.grid(False)
    ay.tick_params(labelsize=12)
    ay.xaxis.pane.fill = False # Left pane
    ay.yaxis.pane.fill = False # Right pane
    ay.zaxis.pane.fill = False # bottom pane
    ay.set_zticks([])
    # # Transparent spines
    # ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    # ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ay.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    # # Transparent panes
    ay.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ay.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ay.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    plt.show()


def offsetwaves(path, color_codes):
    hf = h5py.File(path, 'r')
    freqs = np.array(hf.get('frequencies'))
    X_asd = np.array(hf.get('XASD RMS Avg'))
    Y_asd = np.array(hf.get('YASD RMS Avg'))
    
    #mpl.rcParams['figure.dpi'] = 1200

    
    fig0, ax0 = plt.subplots()
    fig1, ax1 = plt.subplots()
    legendlist = []
    offset = np.linspace(0,10,len(X_asd))
    offsety = np.linspace(0,10,len(X_asd))
    for i in range(len(X_asd)):
        ax0.semilogy(freqs, X_asd[i,:]**2 , color=color_codes[i]) # + offset[i]
        ax1.semilogy(freqs, Y_asd[i,:]**2  , color=color_codes[i]) #+ offsety[i]

        legendlist.append('Sphere ' + str(i))
    
    ax0.set_xlim(60,200)
    # ax0.set_ylim(0,6.5E-15)
    ax0.set_xlabel('Frequency (Hz)', fontsize=18)
    ax0.set_ylabel('Sphere Index', fontsize=18)
    spherenames = [str(x+1) for x in range(25)]
    #ax0.set_yticks(offset, labels=spherenames)
    #ax0.legend(legendlist, fontsize=12, bbox_to_anchor=(1.04, 0), loc="lower left", borderaxespad=0)
    ax0.set_title('X Motion Spectra', fontsize=22)
    
    ax1.set_xlim(60,150)
    # ax1.set_ylim(0,3.2E-14)
    ax1.set_xlabel('Frequency (Hz)', fontsize=18)
    ax1.set_ylabel('Sphere Index', fontsize=18)
    #ax1.set_yticks(offsety, labels=spherenames)
    #ax1.legend(legendlist, fontsize=12, bbox_to_anchor=(1.04, 0), loc="lower left", borderaxespad=0)
    ax1.set_title('Y Motion Spectra', fontsize=22)

    plt.show()

def justpeakplotter(path, color_codes):
    hf = h5py.File(path, 'r')
    freqs = np.array(hf.get('frequencies'))
    X_asd = np.array(hf.get('XASD RMS Avg'))
    Y_asd = np.array(hf.get('YASD RMS Avg'))
    
    #mpl.rcParams['figure.dpi'] = 1200
    
    x_peaks_list = [[] for i in range(len(X_asd))]
    y_peaks_list = [[] for i in range(len(X_asd))]
    

    legendlist = []

    for i in range(len(X_asd)):
        x_peak_indices, x_peak_dict = find_peaks(X_asd[i,:], height=3E-9)
        x_peak_heights = x_peak_dict['peak_heights']
        x_peak_freqs = freqs[x_peak_indices]
        
        x_peaks = (np.vstack((x_peak_indices, x_peak_freqs, x_peak_heights))).T

        
        x_peaks_list[i] = x_peaks
        
        y_peak_indices, y_peak_dict = find_peaks(Y_asd[i,:], height=3E-9, threshold=1E-11)
        y_peak_heights = y_peak_dict['peak_heights']
        y_peak_freqs = freqs[y_peak_indices]
        
        y_peaks = (np.vstack((y_peak_indices, y_peak_freqs, y_peak_heights))).T
        y_peaks_list[i] = y_peaks
        
        legendlist.append('Sphere ' + str(i))
    
        
    # highest_peak_index = peak_indices[np.argmax(peak_heights)]
    # second_highest_peak_index = peak_indices[np.argpartition(peak_heights,-2)[-2]]
    
    fig, ax = plt.subplots()
    
    for i in range(len(x_peaks_list)):
        sortedpeaksind = np.argsort(x_peaks_list[i][:,2])[::-1]
        numpeaks = 2
        counter = 0
        j=0
        while counter < numpeaks:
            peakind = sortedpeaksind[j]
            if x_peaks_list[i][peakind,1] > 60:
                X_asdi = X_asd[i]
                lb = int(x_peaks_list[i][peakind,0]-10)
                up = int(x_peaks_list[i][peakind,0]+10)
                ax.semilogy(freqs[lb:up], X_asdi[lb:up], color=color_codes[i], label =('Sphere ' + str(i+1)))
                
                counter+=1
            j+=1

                
    ax.grid()
    #ax.set_xlim(0,170)
    ax.set_xlabel('Frequency (Hz)', fontsize=18)
    ax.set_ylabel(r'ASD ($m/ \sqrt{Hz}$)', fontsize=18)
    
    handles, labels = gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=12, bbox_to_anchor=(1.04, 0), loc="lower left", borderaxespad=0)
    
    ax.set_title('X motion ASD', fontsize=22)
    
    
    figa, axa = plt.subplots()
    
    for i in range(len(y_peaks_list)):
        sortedpeaksind = np.argsort(y_peaks_list[i][:,2])[::-1]
        numpeaks = 2
        counter = 0
        j=0
        while counter < numpeaks:
            if j < len(sortedpeaksind):
                peakind = sortedpeaksind[j]
                if y_peaks_list[i][peakind,1] > 60:
                    Y_asdi = Y_asd[i]
                    lb = int(y_peaks_list[i][peakind,0]-10)
                    up = int(y_peaks_list[i][peakind,0]+10)
                    axa.semilogy(freqs[lb:up], Y_asdi[lb:up], color=color_codes[i], label =('Sphere ' + str(i+1)))
                    
                    counter+=1
                j+=1
            else:
                break
                
    axa.grid()
    #axa.set_xlim(0,170)
    axa.set_xlabel('Frequency (Hz)', fontsize=18)
    axa.set_ylabel(r'ASD ($m/ \sqrt{Hz}$)', fontsize=18)
    
    handles, labels = gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axa.legend(by_label.values(), by_label.keys(), fontsize=12, bbox_to_anchor=(1.04, 0), loc="lower left", borderaxespad=0)
    
    axa.set_title('Y motion ASD', fontsize=22)

    plt.show()
    
###############################################################################
    
totalspheres = 25
file = r'D:\Lab data\20240604\0-8MHz\0-8MHz_rmsavg.h5'
color_map = mpl.colormaps.get_cmap('CMRmap')
colorind = np.linspace(0,0.8,totalspheres)
color_codes = [color_map(colorind[i]) for i in range(totalspheres)]

multisphere3d(file, color_codes[::-1])