

import pandas as pd
import numpy as np
import scipy as sp
import math
import matplotlib.pyplot as plt
import os
from scipy import signal

#
path = r"D:\Lab data\20250131"
os.chdir(path)

aodinc = 2E-6
lazinc = 8e-7

filename = 'lazno.csv'
lazdf = pd.read_csv(filename, delimiter=',', header=2, usecols=[0,1,2,3], names=['Time','X','Y','Sum'])
lazdf.Time = lazdf.Time*lazinc

filename = 'aod2.csv'
aoddf = pd.read_csv(filename, delimiter=',', header=2, usecols=[0,1,2,3], names=['Time','X','Y','Sum'])
aoddf.Time = aoddf.Time*aodinc

fig, ax = plt.subplots()

for col in lazdf.columns:
    if col != 'Time':
        print(len(lazdf[col]))
        f,psd = signal.welch(lazdf[col],1/lazinc,nperseg=len(lazdf[col]))
        ax.loglog(f,psd,label='Laser '+ col)


for col in aoddf.columns:
    if col != 'Time':
        print(len(aoddf[col]))
        f,psd = signal.welch(aoddf[col],1/aodinc,nperseg=len(aoddf[col]))
        ax.loglog(f,psd,label='AOD '+col,alpha=0.3)
ax.set_ylim(1e-11,1e-5)
ax.legend()
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('PSD (V^2/Hz)')
ax.grid(True,which='major', linestyle='-')
ax.grid(True,which='minor', linestyle='--',alpha=0.3)
plt.show()


