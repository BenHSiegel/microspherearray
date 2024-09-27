"""
Created on Thu May 16 11:42:25 2024
Correlation matrix generator for spheres
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



def butter_bandpass(data, highpassfq, fs, order=3):
    '''
    Passes data through a bandpass filter with a variable high pass corner and
    set low pass corner of 250 Hz
    Inputs:
        data = timestream to filter
        highpassfq = high pass frequency corner
        fs = sampling frequency
        order = order of butterworth filter (default 3)
    Outputs:
        filtered_data = timestream of data after filtering
    '''
    #declare nyquist frequency
    nyq = 0.5 * fs
    #set high and low pass corners
    highpasscornerfq = highpassfq / nyq
    lowpasscornerfq = 250/nyq
    #create bandpass butterworth filter
    b, a = butter(order, [highpasscornerfq, lowpasscornerfq], btype='bandpass')
    #pass data through filter
    filtered_data = lfilter(b, a, data)

    return filtered_data

    
def corr_cross_calc(X, Y):
    '''
    calculates the Pearson correlation coefficient between the columns of two 
    arrays for all combinations of the columns
    Inputs:
        X = first data array
        Y = second data array
    Outputs:
        cor_matrix = grid of correlation coefficients where the n,m element
        is the nth column of X crossed with the mth column of Y
    '''
    cor_matrix = np.zeros((X.shape[1],Y.shape[1]))
    
    for n in range(X.shape[1]):
        for m in range(Y.shape[1]):
            cor_matrix[n,m] = pearsonr(X[:,n], Y[:,m])[0]
    
    return cor_matrix



    
def hdf5file_correlationprocessing(path, totalspheres, sep, saveflag, savename):
    hdf5_list = []
    for filename in sorted(os.listdir(path)):
        if filename.endswith(".h5") and not filename.endswith("avg.h5") and not filename.endswith("matrix.h5") :
            hdf5_list.append(filename)
    
    xcorrlist = []
    ycorrlist = []
    xycorrlist = []
    xcross_SD_list = []
    ycross_SD_list = []
    fftbinning = 2048
    

    counter = 0
    for i in hdf5_list:
        hf = h5py.File(i, 'r')
        group = hf.get('position')
        fs = group.attrs['framerate (fps)']

        segmentsize = int(round(fs/40)*10)
        overlap = int(round(segmentsize/40)*10)
    
        xposdata = []
        yposdata = []
        xfiltdata = []
        yfiltdata = []

        l = 0
        for j in group.items():
            pos = np.array(j[1])
            xpos = pos[:,1].reshape(-1,1)
            xfiltered = butter_bandpass(xpos, 40, fs)
            ypos = pos[:,2].reshape(-1,1)
            yfiltered = butter_bandpass(ypos, 40, fs)
            if l == 0:
                xfiltdata = xfiltered[:,0].reshape(-1,1)
                yfiltdata = yfiltered[:,0].reshape(-1,1)
                xposdata = xpos[:,0].reshape(-1,1)
                yposdata = ypos[:,0].reshape(-1,1)

            else:
                xfiltdata = np.concatenate((xfiltdata, xfiltered[:,0].reshape(-1,1)), axis=1)
                yfiltdata = np.concatenate((yfiltdata, yfiltered[:,0].reshape(-1,1)), axis=1)
                xposdata = np.concatenate((xposdata, xpos[:,0].reshape(-1,1)), axis=1)
                yposdata = np.concatenate((yposdata, ypos[:,0].reshape(-1,1)), axis=1)
            l+=1
            
        hf.close()
        
        xdf = pd.DataFrame(xfiltdata)
        ydf = pd.DataFrame(yfiltdata)
        
        xcorrmatrix = xdf.corr()
        ycorrmatrix = ydf.corr()  
        xycorrmatrix = corr_cross_calc(xfiltdata, yfiltdata)
        
        coherfreq = coherence(xposdata[:,0],xposdata[:,1], fs = fs, window='hann', nperseg=segmentsize, noverlap=overlap, nfft=fftbinning)[0]

        xcohermatrix = [ [ coherence(xposdata[:,m],xposdata[:,n], fs, nperseg=segmentsize, noverlap=overlap, nfft=fftbinning)[1] for n in range(totalspheres) ] for m in range(totalspheres) ]
        ycohermatrix = [ [ coherence(yposdata[:,m],yposdata[:,n], fs, nperseg=segmentsize, noverlap=overlap, nfft=fftbinning)[1] for n in range(totalspheres) ] for m in range(totalspheres) ]


        if counter == 0:
            xcorrlist = xcorrmatrix
            ycorrlist = ycorrmatrix
            xycorrlist = xycorrmatrix
            xcross_SD_list = xcohermatrix
            ycross_SD_list = ycohermatrix

        else:
            xcorrlist = np.dstack((xcorrlist,xcorrmatrix))
            ycorrlist = np.dstack((ycorrlist,ycorrmatrix))
            xycorrlist = np.dstack((xycorrlist, xycorrmatrix))
            
            for m in range(totalspheres):
                for n in range(totalspheres):
                    xcross_SD_list[m][n] = np.mean(np.stack((xcohermatrix[m][n], coherence(xposdata[:,m], xposdata[:,n], fs, nperseg=segmentsize, noverlap=overlap, nfft=fftbinning)[1] )), axis=0)
                    ycross_SD_list[m][n] = np.mean(np.stack((ycohermatrix[m][n], coherence(yposdata[:,m], yposdata[:,n], fs, nperseg=segmentsize, noverlap=overlap, nfft=fftbinning)[1] )), axis=0)

        counter += 1

    if len(hdf5_list) == 1:
        xcorr_averaged = xcorrlist
        ycorr_averaged = ycorrlist
        xycorr_averaged = xycorrlist
    else:        
        xcorr_averaged = np.mean(xcorrlist, axis=2)
        ycorr_averaged = np.mean(ycorrlist, axis=2)
        xycorr_averaged = np.mean(xycorrlist, axis=2)
        
    
    if saveflag:
        savename = savename + '.h5'
        hf = h5py.File(savename, 'w')
        hf.attrs.create('framerate (fps)', fs)
        hf.attrs.create('Separation (MHz)', sep)
        hf.attrs.create('Number of spheres', totalspheres)
        hf.create_dataset('X_correlations', data=xcorr_averaged)
        hf.create_dataset('Y_correlations', data=ycorr_averaged)
        hf.create_dataset('X-Y Cross Correlations', data=xycorr_averaged)
        g1 = hf.create_group('X CSDs')
        g1.attrs.create("Indexing Info", 'Row 0 is the frequency bins, Row n is the CSD of sphere x with sphere n')
        g2 = hf.create_group('Y CSDs')
        g2.attrs.create("Indexing Info", 'Row 0 is the frequency bins, Row n is the CSD of sphere x with sphere n')
        for m in range(totalspheres):
            for n in range(totalspheres):
                if n ==0:
                    xcsdi = xcross_SD_list[m][n]
                    ycsdi = ycross_SD_list[m][n]
                else:
                    xcsdi = np.concatenate((xcsdi, xcross_SD_list[m][n]),axis=0)
                    ycsdi = np.concatenate((ycsdi, ycross_SD_list[m][n]),axis=0)
            xcsdm = np.concatenate((coherfreq, xcsdi),axis=0)
            ycsdm = np.concatenate((coherfreq, ycsdi),axis=0)
            g1.create_dataset('Sphere '+ str(m+1),data=xcsdm)
            g2.create_dataset('Sphere '+ str(m+1),data=ycsdm)

        hf.close()
        
    
    return xcorr_averaged, ycorr_averaged, xycorr_averaged, xcross_SD_list, ycross_SD_list, coherfreq


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
    ax.tick_params(top=False, bottom=True,
                   labeltop=False, labelbottom=True)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=0, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=0)
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


def folder_walker_correlation_calc(main_directory, totalspheres, saveflag, savefigs):

    counter = 0
    correlation_scan = []
    xasddata = [ [] for i in range(totalspheres)]
    yasddata = [ [] for i in range(totalspheres)]
    freqasddata = [ [] for i in range(totalspheres)]
    separation_scan = []
    x_peak_scan = []
    y_peak_scan = []

    figcros = plt.figure()
    axcros = figcros.add_subplot(111,projection='3d')
    xcros = ycros = np.arange(0,totalspheres,1)

    for path, folders, files in os.walk(main_directory):
       
        for folder_name in folders:
            
            directory = f"{path}/{folder_name}"
            os.chdir(directory)
            print(folder_name)
            with open('info.txt') as file:
                lines = [line.rstrip() for line in file]
            framerate = float(lines[0])
            sep = float(lines[1])
            umsep = sep * 70
            data_label = lines[2]
            include_in_scan = lines[3]
            
            if include_in_scan == 'True':
                counter += 1
                
            savename = str(sep) + 'correlationmatrix'
            xcorr_averaged, ycorr_averaged, xycorr_averaged, xcross_SD_list, ycross_SD_list, coherfreq = hdf5file_correlationprocessing(directory, totalspheres, sep, saveflag, savename)

            figcsd, axcsd = plt.subplots(2)
            figcsd.suptitle('Cross Spectral Density for ' + str(umsep) + r'$\mu m$ Spacing')
            axcsd[0].set_title('X CSDs')
            axcsd[1].set_title('Y CSDs')
            axcsd[0].set_xlabel('Frequency (Hz)')
            axcsd[0].set_ylabel(r'CSD (${m}^2/Hz$)')
            axcsd[1].set_xlabel('Frequency (Hz)')
            axcsd[1].set_ylabel(r'CSD (${m}^2/Hz$)')

            for m in range(len(xcross_SD_list)):
                for n in range(len(xcross_SD_list)):
                    if m!=n:
                        label = str(m) + '-' + str(n)
                        axcsd[0].plot(coherfreq, xcross_SD_list[m][n], label=label)
                        # x_peak_indices, x_peak_dict = find_peaks(xcross_SD_list[m][n])
                        
                        # x_peak_heights = x_peak_dict['peak_heights']
                        # sortedind = np.argsort(x_peak_heights)[::-1]
                        # sorted_x_peak_indices = [x_peak_indices[i] for i in sortedind]
                        # x_peak_freqs = coherfreq[x_peak_indices[sorted_x_peak_indices]]


                        axcsd[1].plot(coherfreq, ycross_SD_list[m][n], label=label)

            handles, labels = axcsd[0].get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            axcsd[0].legend(by_label.values(), by_label.keys(), fontsize=12, loc="lower right", borderaxespad=1)

            handles, labels = axcsd[1].get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            axcsd[1].legend(by_label.values(), by_label.keys(), fontsize=12, loc="lower right", borderaxespad=1)

            plt.close(figcsd)
            
            if include_in_scan == 'True':
                
                xcorr_offdiags = []
                ycorr_offdiags = []
                for i in range(np.shape(xcorr_averaged)[0]):
                    for j in range(np.shape(xcorr_averaged)[1]):
                        if j > i:
                            xcorr_offdiags.append(xcorr_averaged[i][j])
                            ycorr_offdiags.append(xcorr_averaged[i][j])
            
                correlation_scan.append((np.vstack((xcorr_offdiags, ycorr_offdiags))).T)
                
                xgrid, ygrid = np.meshgrid(xcros, ycros)
                zgrid = np.full((totalspheres,totalspheres), umsep)
                xycorr_abs = np.abs(xycorr_averaged)
                c_min, c_max = 0, 0.25
                norm = mpl.colors.Normalize(vmin=c_min, vmax=c_max)
                color_map = mpl.cm.viridis
                axcros.plot_surface(xgrid, ygrid, zgrid, rstride=1, cstride=1, facecolors = color_map(norm(xycorr_abs)), shade=False)

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
                    
                    
                    if counter == 1:
                        freqasddata[i] = freq.reshape(-1,1)
                        xasddata[i] = xpsd[i,:].reshape(-1,1)
                        yasddata[i] = ypsd[i,:].reshape(-1,1)
                        
                        
                    else:
                        freqasddata[i] = np.concatenate((freqasddata[i], freq.reshape(-1,1)), axis=1)
                        xasddata[i] = np.concatenate((xasddata[i], xpsd[i,:].reshape(-1,1)), axis=1)
                        yasddata[i] = np.concatenate((yasddata[i], ypsd[i,:].reshape(-1,1)), axis=1)
                  
            
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
            
            if savefigs:
                fig.savefig(os.path.join(main_directory, data_label + '.png'))   # save the figure to file
            
            
            plt.close(fig)
            
            

        break

    axcros.set_xlabel('Sphere Index')
    axcros.set_ylabel('Sphere Index')
    axcros.set_zlabel('Intersphere Separation (um)')
    axcros.set_title('X-Y Correlations at Varying Distances')
    cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=color_map), ax=axcros, shrink=0.6, label="Correlation")

    return x_peak_scan, y_peak_scan, separation_scan, correlation_scan, freqasddata, xasddata, yasddata, xcross_SD_list, ycross_SD_list, coherfreq


def plot_correlations_vs_separations(x_peak_scan, y_peak_scan, separation_scan, correlation_scan, main_directory, totalspheres, savefigs, color_codes):

    figa, axa = plt.subplots(1,2, tight_layout=True)
    #figa.set_size_inches(10,5)
    #figa.set_dpi(600)

    figb, axb = plt.subplots(1,2, tight_layout=True)
    #figb.set_size_inches(10,5)
    #figb.set_dpi(600)

    figc, axc = plt.subplots(1,2, tight_layout=True)
    #figc.set_size_inches(10,5)
    #figc.set_dpi(600)

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
    axb[0].legend(by_label.values(), by_label.keys(), fontsize=12, loc="lower right", borderaxespad=1)

    handles, labels = axc[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axc[0].legend(by_label.values(), by_label.keys(), fontsize=12, borderaxespad=1)

    if savefigs:
        figa.savefig(os.path.join(main_directory, 'correlation.png'))   # save the figure to file
        
        figb.savefig(os.path.join(main_directory, 'freqshift.png'))   # save the figure to file
        
        figc.savefig(os.path.join(main_directory, 'amplitudeshift.png'))   # save the figure to file
    plt.show()
    #plt.close('all')
    return


def plot_separation_ASD_scan(freqasddata, xasddata, yasddata, separation_scan, main_directory, savefigs, color_codes):

    figs={}
    axs={}
    for i in range(len(freqasddata)):
        figs[i], axs[i] = plt.subplots(2, 1, sharex=True, tight_layout=True)
        figs[i].set_size_inches(18.5, 10.5)
        figs[i].set_dpi(800)
        
        alpharange = np.linspace(0.1, 1, (freqasddata[1].shape)[1])[::-1]
        for j in range((freqasddata[i].shape)[1]):
            label_name = str(separation_scan[j]) + 'um'
            axs[i][0].semilogy(freqasddata[i][:,j], xasddata[i][:,j], color = color_codes[i], alpha = alpharange[j], label=label_name)
            axs[i][1].semilogy(freqasddata[i][:,j], yasddata[i][:,j], color = color_codes[i], alpha = alpharange[j], label=label_name)
        axs[i][0].set_xlim([5,350])
        axs[i][1].set_xlim([5,350])

        handles, labels = axs[i][0].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        axs[i][0].legend(by_label.values(), by_label.keys(), fontsize=12, loc="upper right", bbox_to_anchor=(1.1, 1), borderaxespad=0.1)
        titlename = "Sphere " + str(i) + ' Response'
        figs[i].suptitle(titlename)
        
        axs[i][1].set_xlabel('Frequency (Hz)')
        axs[i][0].set_ylabel(r'X ASD ($m/ \sqrt{Hz}$)')
        axs[i][1].set_ylabel(r'Y ASD ($m/ \sqrt{Hz}$)')
        
        if savefigs:
            figs[i].savefig(os.path.join(main_directory, titlename +'.png'))

    #plt.close('all')
    return


def interpdatafn(freq, data, lb, up, separation_scan, sphrindex, direction, fig, ax):
    
    datapoints = max(len(x) for x in freq.T)
    freqlist = np.linspace(lb, up, datapoints)
    interpdata = []
    counter = 0
    for i in range(len(data.T)):
        f = interp1d(freq[:,i],data[:,i])
        amp = f(freqlist)
        normamp = amp / max(amp)
        if counter == 0:
            interpdata = amp.reshape(-1,1)
        else:
            interpdata = np.concatenate((interpdata, amp.reshape(-1,1)), axis=1)
        counter += 1

    df = pd.DataFrame(data=interpdata, columns = np.round(separation_scan).astype(int))
    num_ticks = 10
    ylist = freqlist.tolist()
    yticks = np.linspace(0, len(ylist)-1, num_ticks, dtype=int)
    yticklabel = [np.round(ylist[idx]).astype(int) for idx in yticks]
    sn.heatmap(df, ax=ax[sphrindex], norm = LogNorm(), yticklabels=yticklabel, cbar_kws={'label': r'Amplitude ($m/ \sqrt{Hz}$)'})
    ax[sphrindex].invert_yaxis()
    ax[sphrindex].set_yticks(yticks)
    ax[sphrindex].set_yticklabels(yticklabel)
    titlename = "Sphere " + str(sphrindex) + direction + 'Response'
    ax[sphrindex].set_title(titlename)
    #ax[sphrindex].set_ylabel('Frequency (Hz)')
    ax[sphrindex].set_xlabel(r'Separation ($\mu m$)')
    
    return freqlist, interpdata, df, fig, ax

def heatmap_scan_plotter(freqasddata, xasddata, yasddata, anticrossinglbs, anticrossingubs, separation_scan, main_directory, totalspheres, savefigs):
    figx, axx = plt.subplots(1, totalspheres, figsize=(totalspheres*5, 5), sharex=False, tight_layout=True)
    figx.set_dpi(800)
    figy, axy = plt.subplots(1, totalspheres, figsize=(totalspheres*5, 5), sharex=False, tight_layout=True)
    figy.set_dpi(800)
    for i in range(totalspheres):
        freqlistx, interpdatax, dfx, figx, axx = interpdatafn(freqasddata[i], xasddata[i], anticrossinglbs, anticrossingubs, separation_scan, i, " X ", figx, axx)
        freqlisty, interpdatay, dfy, figy, axy = interpdatafn(freqasddata[i], yasddata[i], anticrossinglbs, anticrossingubs, separation_scan, i, " Y ", figy, axy)
    axx[0].set_ylabel('Frequency (Hz)')
    axy[0].set_ylabel('Frequency (Hz)')
    if savefigs:
        savenamex = "X fq vs separation"
        savenamey = "Y fq vs separation"
        figx.savefig(os.path.join(main_directory, savenamex +'.png'))
        figy.savefig(os.path.join(main_directory, savenamey +'.png'))


    #plt.close('all')
    return



# main_directory = r"D:\Lab data\20240905\New folder"
# totalspheres = 2
# saveflag = True
# savefigs = True
# anticrossinglbs = 60
# anticrossingubs = 300

# color_value = np.linspace(0,1,totalspheres)
# color_value_T = color_value[::-1]
# color_codes = [(color_value[i],0,color_value_T[i]) for i in range(totalspheres)]

# x_peak_scan, y_peak_scan, separation_scan, correlation_scan, freqasddata, xasddata, yasddata, xcross_SD_list, ycross_SD_list, coherfreq = folder_walker_correlation_calc(main_directory, totalspheres, saveflag, savefigs)
# #plot_correlations_vs_separations(x_peak_scan, y_peak_scan, separation_scan, correlation_scan, main_directory, totalspheres, savefigs, color_codes)
# #plot_separation_ASD_scan(freqasddata, xasddata, yasddata, separation_scan, main_directory, savefigs, color_codes)
# #heatmap_scan_plotter(freqasddata, xasddata, yasddata,  anticrossinglbs, anticrossingubs, separation_scan, main_directory, totalspheres, savefigs)