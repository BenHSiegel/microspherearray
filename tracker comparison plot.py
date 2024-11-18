'''
Tracker comparison plot maker
'''
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
from scipy.ndimage import variance
from scipy.signal import welch
from scipy.signal.windows import blackman
from scipy.signal import find_peaks, butter, lfilter, csd, coherence
from matplotlib.pyplot import gca
import h5py
import pandas as pd
from matplotlib.colors import LogNorm
import seaborn as sn
from scipy.interpolate import interp1d
import matplotlib.pylab as pl
import matplotlib.gridspec as gridspec
from basichdf5 import hdf5_sphere_data_scraper as scraper


path = r'C:\Users\Ben\Documents\Research\Tracker comparison'
os.chdir(path)
x_trackpy, y_trackpy, _, _, _, _ = scraper('ulp1.h5')