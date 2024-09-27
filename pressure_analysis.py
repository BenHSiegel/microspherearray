'''
Script to run through hdf5 files containing data of one sphere's motion at different
pressures with 1 trapping beam, 2 trapping beams with the second dark, and 2 trapping
beams with the second bright.
This is to analyze if the timesharing leads to heating of the sphere's motion.
'''


import os
import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import fft
from sklearn.decomposition import PCA
from matplotlib.mlab import psd
from scipy.optimize import minimize, curve_fit
from scipy.stats import chi2
from scipy.signal import welch
from scipy.signal.windows import blackman
from scipy.signal import find_peaks
from matplotlib.pyplot import gca
from basichdf5 import hdf5_sphere_data_scraper

def lorentzian(f, f_0, T, gamma):
    kb = 1.38e-23 # Boltzmann's constant, SI units
    m = 1e-12 # mass in kg
    omega = 2*np.pi*f
    omega0 = 2*np.pi*f_0
    return kb*T/(np.pi * m) * gamma/((omega0**2 - omega**2)**2 + omega**2 * gamma**2)

path = r'D:\Lab data\20240919'
os.chdir(path)

file_name_directory = []
for filename in sorted(os.listdir(path)):
    if filename.endswith(".h5") and not filename.endswith("mod.h5") and not filename.endswith("matrix.h5"):
        file_name_directory.append(filename)

color_map = mpl.colormaps.get_cmap('CMRmap')
pressureind = int(len(file_name_directory)/3)
colorind = np.linspace(0,0.8,pressureind)
color_codes = [color_map(colorind[i]) for i in range(pressureind)]


mpl.rcParams.update({"axes.grid" : True})

figa, axa = plt.subplots(2, sharex=True)
figa.suptitle('One beam on data')
axa[0].set_title('X Motion')
axa[1].set_title('Y Motion')
axa[1].set_xlabel('Frequency (Hz)')
axa[0].set_ylabel(r'ASD ($m/\sqrt{Hz}$)')
axa[1].set_ylabel(r'ASD ($m/\sqrt{Hz}$)')

figb, axb = plt.subplots(2, sharex=True)
figb.suptitle('Two beams, second dark')
axb[0].set_title('X Motion')
axb[1].set_title('Y Motion')
axb[1].set_xlabel('Frequency (Hz)')
axb[0].set_ylabel(r'ASD ($m/\sqrt{Hz}$)')
axb[1].set_ylabel(r'ASD ($m/\sqrt{Hz}$)')

figc, axc = plt.subplots(2, sharex=True)
figc.suptitle('Two beams, second bright')
axc[0].set_title('X Motion')
axc[1].set_title('Y Motion')
axc[1].set_xlabel('Frequency (Hz)')
axc[0].set_ylabel(r'ASD ($m/\sqrt{Hz}$)')
axc[1].set_ylabel(r'ASD ($m/\sqrt{Hz}$)')



single_paramsx = []
dualoff_paramsx = []
dualon_paramsx = []
single_paramsy = []
dualoff_paramsy = []
dualon_paramsy = []

i1=i2=i3=0
for pressurefile in file_name_directory:
    xposdata, yposdata, xfftmatrix, yfftmatrix, frequency_bins, fs = hdf5_sphere_data_scraper(pressurefile)

    if pressurefile[0] == '1':
        pressure = pressurefile[7:-7]
        pressure = pressure.replace('_','.')
        axa[0].semilogy(frequency_bins, xfftmatrix, color=color_codes[i1], label=pressure)
        axa[1].semilogy(frequency_bins, yfftmatrix, color=color_codes[i1], label=pressure)
        
        xinit_guess = [150, 10000, 700]
        yinit_guess = [150, 10000, 700]
        #with current binning of 1024 frequencies and sampling rate of 1000
        xfmin = int(150 / 0.488) # 0.488 converts the frequency to the frequency index
        xfmax = int(250 / 0.488)
        xfreqfit = frequency_bins[xfmin:xfmax]
        xpsdfit = np.square(xfftmatrix[xfmin:xfmax,0])
        fit_params, cov = curve_fit(lorentzian, xfreqfit, xpsdfit, p0=xinit_guess)#, bounds=([100,50,20],[300,1000,150])

        single_paramsx.append(fit_params)

        #with current binning of 1024 frequencies and sampling rate of 1000
        yfmin = int(100 / 0.488) # 0.488 converts the frequency to the frequency index
        yfmax = int(200 / 0.488)
        yfreqfit = frequency_bins[yfmin:yfmax]
        ypsdfit = np.square(yfftmatrix[yfmin:yfmax,0])
        fit_params, cov = curve_fit(lorentzian, yfreqfit, ypsdfit, p0=yinit_guess)#, bounds=([100,50,20],[300,1000,150])

        single_paramsy.append(fit_params)

        i1 += 1
    
    if pressurefile[0] == '2' and pressurefile[6] == 'f':
        pressure = pressurefile[9:-7]
        pressure = pressure.replace('_','.')
        axb[0].semilogy(frequency_bins, xfftmatrix, color=color_codes[i2], label=pressure)
        axb[1].semilogy(frequency_bins, yfftmatrix, color=color_codes[i2], label=pressure)

        xinit_guess = [200, 295, 100]
        yinit_guess = [150, 295, 100]
        #with current binning of 1024 frequencies and sampling rate of 1000
        xfmin = int(150 / 0.488) # 0.488 converts the frequency to the frequency index
        xfmax = int(250 / 0.488)
        xfreqfit = frequency_bins[xfmin:xfmax]
        xpsdfit = np.square(xfftmatrix[xfmin:xfmax,0])
        fit_params, cov = curve_fit(lorentzian, xfreqfit, xpsdfit, p0=xinit_guess)#, bounds=([100,50,20],[300,1000,150])

        dualoff_paramsx.append(fit_params)

        #with current binning of 1024 frequencies and sampling rate of 1000
        yfmin = int(100 / 0.488) # 0.488 converts the frequency to the frequency index
        yfmax = int(200 / 0.488)
        yfreqfit = frequency_bins[yfmin:yfmax]
        ypsdfit = np.square(yfftmatrix[yfmin:yfmax,0])
        fit_params, cov = curve_fit(lorentzian, yfreqfit, ypsdfit, p0=yinit_guess)#, bounds=([100,50,20],[300,1000,150])

        dualoff_paramsy.append(fit_params)

        i2 +=1

    if pressurefile[0] == '2' and pressurefile[6] == 'n':
        pressure = pressurefile[8:-7]
        pressure = pressure.replace('_','.')
        axc[0].semilogy(frequency_bins, xfftmatrix, color=color_codes[i3], label=pressure)
        axc[1].semilogy(frequency_bins, yfftmatrix, color=color_codes[i3], label=pressure)

        xinit_guess = [200, 295, 100]
        yinit_guess = [150, 295, 100]
        #with current binning of 1024 frequencies and sampling rate of 1000
        xfmin = int(150 / 0.488) # 0.488 converts the frequency to the frequency index
        xfmax = int(250 / 0.488)
        xfreqfit = frequency_bins[xfmin:xfmax]
        xpsdfit = np.square(xfftmatrix[xfmin:xfmax,0])
        fit_params, cov = curve_fit(lorentzian, xfreqfit, xpsdfit, p0=xinit_guess)#, bounds=([100,50,0],[300,1000,500])

        dualon_paramsx.append(fit_params)

        #with current binning of 1024 frequencies and sampling rate of 1000
        yfmin = int(100 / 0.488) # 0.488 converts the frequency to the frequency index
        yfmax = int(200 / 0.488)
        yfreqfit = frequency_bins[yfmin:yfmax]
        ypsdfit = np.square(yfftmatrix[yfmin:yfmax,0])
        fit_params, cov = curve_fit(lorentzian, yfreqfit, ypsdfit, p0=yinit_guess)#, bounds=([100,50,20],[300,1000,150])

        dualon_paramsy.append(fit_params)

        i3+=1
    

handles, labels = gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
axa[0].legend(by_label.values(), by_label.keys(), fontsize=12, bbox_to_anchor=(1.04, 0), loc="upper left", borderaxespad=0)
axc[0].legend(by_label.values(), by_label.keys(), fontsize=12, bbox_to_anchor=(1.04, 0), loc="upper left", borderaxespad=0)
axb[0].legend(by_label.values(), by_label.keys(), fontsize=12, bbox_to_anchor=(1.04, 0), loc="upper left", borderaxespad=0)
  

pressurevalues = [0.399, 0.577, 0.702, 0.804, 1.01, 1.23, 1.49, 2.07, 2.87]
figx, axx = plt.subplots(3, sharex=True)
figx.suptitle('X Motion')
figy, axy = plt.subplots(3, sharex=True)
figy.suptitle('Y Motion')
for i in range(len(single_paramsx)):
    axx[0].plot(pressurevalues[i], np.abs(single_paramsx[i][0]), 'b.', label='1beam')
    axx[1].plot(pressurevalues[i], single_paramsx[i][1], 'b.', label='1beam')
    axx[2].plot(pressurevalues[i], single_paramsx[i][2], 'b.', label='1beam')
    
    axy[0].plot(pressurevalues[i], np.abs(single_paramsy[i][0]), 'b.', label='1beam')
    axy[1].plot(pressurevalues[i], single_paramsy[i][1], 'b.', label='1beam')
    axy[2].plot(pressurevalues[i], single_paramsy[i][2], 'b.', label='1beam')

    axx[0].plot(pressurevalues[i], np.abs(dualoff_paramsx[i][0]), 'r.', label='2beamdark')
    axx[1].plot(pressurevalues[i], dualoff_paramsx[i][1], 'r.', label='2beamdark')
    axx[2].plot(pressurevalues[i], dualoff_paramsx[i][2], 'r.', label='2beamdark')
    
    axy[0].plot(pressurevalues[i], np.abs(dualoff_paramsy[i][0]), 'r.', label='2beamdark')
    axy[1].plot(pressurevalues[i], dualoff_paramsy[i][1], 'r.', label='2beamdark')
    axy[2].plot(pressurevalues[i], dualoff_paramsy[i][2], 'r.', label='2beamdark')    

    axx[0].plot(pressurevalues[i], np.abs(dualon_paramsx[i][0]), 'k.', label='2beamdark')
    axx[1].plot(pressurevalues[i], dualon_paramsx[i][1], 'k.', label='2beamdark')
    axx[2].plot(pressurevalues[i], dualon_paramsx[i][2], 'k.', label='2beamdark')
    
    axy[0].plot(pressurevalues[i], np.abs(dualon_paramsy[i][0]), 'k.', label='2beam')
    axy[1].plot(pressurevalues[i], dualon_paramsy[i][1], 'k.', label='2beam')
    axy[2].plot(pressurevalues[i], dualon_paramsy[i][2], 'k.', label='2beam')

axx[0].set_ylabel('Resonant frequency (Hz)')
axx[1].set_ylabel('Temperature')
axx[2].set_ylabel(r'$\gamma$ (Hz)')

axy[0].set_ylabel('Resonant frequency (Hz)')
axy[1].set_ylabel('Temperature')
axy[2].set_ylabel(r'$\gamma$ (Hz)')

axx[2].set_xlabel('Pressure (mbar)')
axy[2].set_xlabel('Pressure (mbar)')

handles, labels = gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
axx[2].legend(by_label.values(), by_label.keys(), fontsize=12, bbox_to_anchor=(1.01, 0), loc="lower left", borderaxespad=0)
axy[2].legend(by_label.values(), by_label.keys(), fontsize=12, bbox_to_anchor=(1.01, 0), loc="lower left", borderaxespad=0)

plt.show()

