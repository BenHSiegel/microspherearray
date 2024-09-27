# -*- coding: utf-8 -*-
"""
Looking at data from qpd
"""

import pandas as pd
import numpy as np
import scipy as sp
import math
import matplotlib.pyplot as plt
import os
from scipy import fft
from scipy import special
from scipy.signal.windows import blackman
from scipy.signal import find_peaks

#
path = r"C:\Users\Ben\Documents\Research\Moore Lab\timeshare qpd data"
os.chdir(path)

inc = 2E-9

filename = 'vppcheck8-24-23.csv'
df = pd.read_csv(filename, delimiter=',', header=2, usecols=[0,1,2], names=['Time','CH1','CH2'])
df.Time = df.Time*inc

fig, ax = plt.subplots()
ax.plot(df.Time,df.CH2)
ax.plot(df.Time,df.CH1)
ax.set_xlim(0,2E-5)

vpp1 = max(df.CH1) - min(df.CH1)
vpp2 = max(df.CH2) - min(df.CH2)
print('Vpp AO0 = ' + str(vpp1))
print('Vpp AO1 = ' + str(vpp2))
