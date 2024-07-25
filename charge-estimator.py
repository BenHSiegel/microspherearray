import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import fft
from sklearn.decomposition import PCA ## need this for principle component analysis below
from matplotlib.mlab import psd
from scipy.optimize import minimize, curve_fit
from scipy.stats import chi2
import os
from scipy.signal import welch
from scipy.signal.windows import blackman
from scipy.signal import find_peaks
from matplotlib.pyplot import gca
import h5py

def hdf5_sphere_data_scraper(filename):
    '''
    Basic hdf5 reader for the sphere data files.
    filename = path to the file you want to read
    Outputs:
    XASD = numpy array where each column in the array is the x amplitude spectral density data from a sphere (in m/root(Hz))
    YASD = numpy array where each column in the array is the y amplitude spectral density data from a sphere (in m/root(Hz))
    freqs = 1D array containing the frequency bins that were used in the PSD calculations
    '''
    #Opens the HDF5 file in read mode
    hf = h5py.File(filename, 'r')
    
    #Reads the different databases to numpy arrays

    freqs = np.array(hf.get('frequencies'))
    X_asd = np.array(hf.get('XASD RMS Avg'))
    Y_asd = np.array(hf.get('YASD RMS Avg'))

    X_psd = np.square(X_asd)
    Y_psd = np.square(Y_asd)

    #close the hdf5 file once complete   
    hf.close()

    return freqs, X_psd, Y_psd


def lorentzian(f, f_0, T, gamma, cfact):
    kb = 1.38e-23 # Boltzmann's constant, SI units
    m = 1e-12 # mass in kg
    omega = 2*np.pi*f
    omega0 = 2*np.pi*f_0
    return 1/(cfact)**2 * 2*kb*T/(np.pi * m) * gamma/((omega0**2 - omega**2)**2 + omega**2 * gamma**2)

def find_T(ref_psd, freqs):

    peak_indices, peak_dict = find_peaks(ref_psd, height=3e-17)
    peak_heights = peak_dict['peak_heights']
    peak_freqs = freqs[peak_indices]
    peaks = (np.vstack((peak_indices, peak_freqs, peak_heights))).T
    sortedpeaksind = np.argsort(peaks[:,2])[::-1]
    j = 0
    while j < len(sortedpeaksind):
        if peaks[sortedpeaksind[j],1] > 60:
            fguess = peaks[sortedpeaksind[j],1]
            aguess = peaks[sortedpeaksind[j],2]
            fguess_ind = peaks[sortedpeaksind[j],0]
            break
        else:
            j += 1

    init_guess = [fguess, 295, 100, 1e-7]
    if fguess_ind + 200 > len(freqs):
        fmax = len(freqs) -1
    else:
        fmax = int(fguess_ind + 200)
    freqfit = freqs[int(fguess_ind-200):fmax]
    ref_psdfit = ref_psd[int(fguess_ind-200):fmax]
    fit_params, cov = curve_fit(lorentzian, freqfit, ref_psdfit, p0=init_guess) #, bounds=([100,50,0,0],[300,1000,np.inf,np.inf])
    
    return fit_params, fguess



reference_motion_file = r"D:\Lab data\20240531\5MHz\5MHz_rmsavg.h5"
charge_motion_file = r"D:\Lab data\20240531\charge check\charge check_rmsavg"


# process the reference files
freqs, X_psd, Y_psd = hdf5_sphere_data_scraper(reference_motion_file)

figx, axx = plt.subplots()
for i in range(X_psd.shape[0]):
    fit_params, f0 = find_T(X_psd[i,:], freqs)
    axx.semilogy(freqs, X_psd[i,:], label='Sphere ' + str(i))
    axx.plot(freqs, lorentzian(freqs, *fit_params), label='Fit ' + str(i))
    print('For x direction of sphere ' + str(i))
    print(f0)
    print(fit_params)
    

figy, axy = plt.subplots()
for i in range(Y_psd.shape[0]):
    fit_params, f0 = find_T(Y_psd[i,:], freqs)
    axy.semilogy(freqs, Y_psd[i,:], label='Sphere ' + str(i))
    axy.plot(freqs, lorentzian(freqs, *fit_params), label='Fit ' + str(i))
    print('For y direction of sphere ' + str(i))
    print(f0)
    print(fit_params)

plt.show()
