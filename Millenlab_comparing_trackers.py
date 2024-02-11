# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 11:18:29 2024

EBC Comparator Script for Generic vs Spatter tracking
@author: bensi
"""

import pandas as pd
import numpy as np
import scipy as sp
import math
import matplotlib.pyplot as plt
import os
from scipy import fft
from scipy import special
from scipy.signal import welch
from scipy.signal.windows import blackman
from scipy.signal import find_peaks
import h5py



##############################################################################

#variables to define for each file
mainpath = r'C:\Users\bensi\Documents\Research\Millen Lab EBC\comparingdata'
os.chdir(mainpath)




##############################################################################


#paths = os.listdir()
# for item in paths:
#     os.chdir(mainpath + item)
    
    
#     data_dir = []
    
#     for filename in sorted(os.listdir(mainpath+item)):
#         if filename.endswith(".csv"):
#             print(filename)

#             data = pd.read_csv(filename, usecols=[3, 4, 7, 8])
            
            
#             file_data_directory.append(data)
#             file_samplerate_directory.append(samplingrate)
            
#             f.close()