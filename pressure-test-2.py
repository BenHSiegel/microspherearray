
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
    if filename.endswith(".h5") and not filename.endswith("mod.h5") and not filename.endswith("modded.h5"):
        file_name_directory.append(filename)

color_map = mpl.colormaps.get_cmap('CMRmap')
pressureind = int(len(file_name_directory))
colorind = np.linspace(0,0.8,pressureind)
color_codes = [color_map(colorind[i]) for i in range(pressureind)]


mpl.rcParams.update({"axes.grid" : True})

figa, axa = plt.subplots(2, sharex=True)
figa.suptitle('Raising Pressure Data')
axa[0].set_title('X Motion')
axa[1].set_title('Y Motion')
axa[1].set_xlabel('Frequency (Hz)')
axa[0].set_ylabel(r'ASD ($m/\sqrt{Hz}$)')
axa[1].set_ylabel(r'ASD ($m/\sqrt{Hz}$)')


i1=i2=i3=0
for pressurefile in file_name_directory:
    xposdata, yposdata, xfftmatrix, yfftmatrix, frequency_bins, fs = hdf5_sphere_data_scraper(pressurefile)

    pressure = pressurefile[:-7]
    axa[0].semilogy(frequency_bins, xfftmatrix, color=color_codes[i1], label=pressure)
    axa[1].semilogy(frequency_bins, yfftmatrix, color=color_codes[i1], label=pressure)
    
    xinit_guess = [200, 295, 100]
    yinit_guess = [150, 295, 100]
    #with current binning of 1024 frequencies and sampling rate of 1000
    xfmin = int(150 / 0.488) # 0.488 converts the frequency to the frequency index
    xfmax = int(250 / 0.488)
    xfreqfit = frequency_bins[xfmin:xfmax]
    xpsdfit = np.square(xfftmatrix[xfmin:xfmax,0])
    #fit_params, cov = curve_fit(lorentzian, xfreqfit, xpsdfit, p0=xinit_guess)#, bounds=([100,50,20],[300,1000,150])

    #single_paramsx.append(fit_params)

    #with current binning of 1024 frequencies and sampling rate of 1000
    yfmin = int(100 / 0.488) # 0.488 converts the frequency to the frequency index
    yfmax = int(200 / 0.488)
    yfreqfit = frequency_bins[yfmin:yfmax]
    ypsdfit = np.square(yfftmatrix[yfmin:yfmax,0])
    #fit_params, cov = curve_fit(lorentzian, yfreqfit, ypsdfit, p0=yinit_guess)#, bounds=([100,50,20],[300,1000,150])

    #single_paramsy.append(fit_params)

    i1 += 1


handles, labels = gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
axa[1].legend(by_label.values(), by_label.keys(), fontsize=12, bbox_to_anchor=(1.01, 0), loc="lower left", borderaxespad=0)
plt.show()



path = r'D:\Lab data\20240919\modulated'
os.chdir(path)

file_name_directory = []
for filename in sorted(os.listdir(path)):
    if filename.endswith(".h5"):
        file_name_directory.append(filename)

fig, ax = plt.subplots(2, sharex=True)
fig.suptitle('Beam Modulation Comparison')
ax[0].set_title('X Motion')
ax[1].set_title('Y Motion')
ax[1].set_xlabel('Frequency (Hz)')
ax[0].set_ylabel(r'ASD ($m/\sqrt{Hz}$)')
ax[1].set_ylabel(r'ASD ($m/\sqrt{Hz}$)')

labels = ['trap', 'prechamber']
i=0
peakx = []
peaky = []
for file in file_name_directory:
    xposdata, yposdata, xfftmatrix, yfftmatrix, frequency_bins, fs = hdf5_sphere_data_scraper(file)

    ax[0].semilogy(frequency_bins, xfftmatrix, label=labels[i])
    ax[1].semilogy(frequency_bins, yfftmatrix, label=labels[i])
    peakx.append(xfftmatrix[170])
    peaky.append(xfftmatrix[170])
    i+=1

handles, labels = gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax[1].legend(by_label.values(), by_label.keys(), fontsize=12, bbox_to_anchor=(1.01, 0), loc="lower left", borderaxespad=0)

print(peakx[1]/peakx[0])
print(peaky[1]/peaky[0])
plt.show()
