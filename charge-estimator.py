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
    print(orig)
    for i in range(orig.shape[1]):
    
        freq, dataPSD_uncor = welch(orig[:,i], framerate, 'hann', segmentsize, segmentsize/4, fftbinning, 'constant', True, 'density', 0,'mean')
        
        PSDlist[i] = dataPSD_uncor
    
    return freq, PSDlist, orig



def hdf5_sphere_psd_scraper(filename):
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



def lorentzian(f, f_0, T, gamma):
    kb = 1.38e-23 # Boltzmann's constant, SI units
    m = 1e-12 # mass in kg
    omega = 2*np.pi*f
    omega0 = 2*np.pi*f_0
    return kb*T/(np.pi * m) * gamma/((omega0**2 - omega**2)**2 + omega**2 * gamma**2)

def Efieldsolver(gamma, f0, amp):
    m = 1e-12 # mass in kg
    V = 0.2* 20 #Vpp
    d = 0.011 #separation of electrode plates in m 
    omega_E = 2 * np.pi * 73 #73 Hz AC drive
    omega0 = 2*np.pi*f0 #resonant frequency of the sphere
    q = amp / ( (V/d /(np.pi * m))**2 * 1/((omega0**2 - omega_E**2)**2 + omega_E**2 * gamma**2))
    return q


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

    init_guess = [fguess, 295, 100]
    if fguess_ind + 200 > len(freqs):
        fmax = len(freqs) -1
    else:
        fmax = int(fguess_ind + 200)
    freqfit = freqs[int(fguess_ind-200):fmax]
    ref_psdfit = ref_psd[int(fguess_ind-200):fmax]
    fit_params, cov = curve_fit(lorentzian, freqfit, ref_psdfit, p0=init_guess)#, bounds=([100,50,20,0],[300,1000,150,np.inf])
    
    return fit_params, fguess



def estimate_charge(charge_psd, freqs, noisefloor):
    power = sum(charge_psd[124:184]) - noisefloor * (freqs[184]-freqs[124])


    return power


reference_motion_file = r"D:\Lab data\20240905\hdf5_datafiles\2_0MHz\2_0MHz_rmsavg.h5"
charge_motion_file = r"D:\Lab data\20240905\hdf5_datafiles\chargecheck\chargecheck_rmsavg.h5"


# process the reference files


#uses the already calculated PSDs
gammas = []
noisefloor = [2E-9, 1.7E-9]
freqs, X_psd, Y_psd = hdf5_sphere_psd_scraper(reference_motion_file)
freqsq, X_charge, Y_charge = hdf5_sphere_psd_scraper(charge_motion_file)
figx, axx = plt.subplots()
for i in range(X_psd.shape[0]):
    fit_params, f0 = find_T(X_psd[i,:], freqs)
    trial_params = [fit_params[0], fit_params[1], fit_params[2]/2]
    axx.plot(freqs, X_psd[i,:], label='Sphere ' + str(i))
    axx.plot(freqs, lorentzian(freqs, *fit_params), label='Fit ' + str(i))
    #axx.plot(freqs, lorentzian(freqs,*trial_params))
    axx.set_xlabel('Frequency (Hz)')
    axx.set_ylabel('ASD (m^2/Hz)')
    print('For x direction of sphere ' + str(i))
    print(f0)
    print(fit_params)
    power = estimate_charge(X_charge[i,:],freqsq, noisefloor[0])
    amp = X_charge[i,150]
    q = Efieldsolver(fit_params[0], fit_params[2], amp)
    print('Charge on sphere '+str(i)+' is '+str(q))
    
    


figy, axy = plt.subplots()
for i in range(Y_psd.shape[0]):
    fit_params, f0 = find_T(Y_psd[i,:], freqs)
    axy.semilogy(freqs, Y_psd[i,:], label='Sphere ' + str(i))
    axy.plot(freqs, lorentzian(freqs, *fit_params), label='Fit ' + str(i))
    axx.set_xlabel('Frequency (Hz)')
    axx.set_ylabel('ASD (m^2/Hz)')
    print('For y direction of sphere ' + str(i))
    print(f0)
    print(fit_params)
    gammas.append(fit_params[2])
plt.show()







'''
#PCA not working well
#Tries principle component analysis to get the peaks separated for getting nice shapes
reference = r"D:\Lab data\20240905\hdf5_datafiles\2_0MHz\lp_highexp.h5"
charge = r'D:\Lab data\20240905\hdf5_datafiles\chargecheck\chargecheck.h5'

x_stack, y_stack, total_stack, framerate = filepositionreader(reference)
sphere1 = np.array([x_stack[:,0], y_stack[:,0]]).transpose()
sphere2 = np.array([x_stack[:,1], y_stack[:,1]]).transpose()
xfrequency, xPSDlist, orig = pcafitting(sphere1, framerate)
yfrequency, yPSDlist, orig = pcafitting(sphere2, framerate)

figx, axx = plt.subplots()
for i in range(len(xPSDlist)):
    #fit_params, f0 = find_T(X_psd[i,:], freqs)
    #trial_params = [fit_params[0], fit_params[1], fit_params[2]/2]
    axx.semilogy(xfrequency, xPSDlist[i], label='Sphere ' + str(i))
    #axx.plot(frequency, lorentzian(freqs, *fit_params), label='Fit ' + str(i))
    #axx.plot(frequency, lorentzian(freqs,*trial_params))
    axx.set_xlabel('Frequency (Hz)')
    axx.set_ylabel('PSD (m^2/Hz)')

figy, axy = plt.subplots()
for i in range(len(yPSDlist)):
    #fit_params, f0 = find_T(Y_psd[i,:], freqs)
    axy.semilogy(yfrequency, yPSDlist[i], label='Sphere ' + str(i))
    #axy.plot(freqs, lorentzian(freqs, *fit_params), label='Fit ' + str(i))
    #axy.set_xlabel('Frequency (Hz)')
    #axy.set_ylabel('ASD (m^2/Hz)')
    # print('For y direction of sphere ' + str(i))
    # print(f0)
    # print(fit_params)

'''