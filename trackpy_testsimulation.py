'''
Testing trackpy vs simulated data
'''
import pandas as pd
import trackpy as tp
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.mlab import psd
from scipy.optimize import minimize, curve_fit
from scipy.stats import chi2
import os
from scipy.signal import welch
from scipy.signal import find_peaks
from matplotlib.pyplot import gca
import h5py

path = r'C:\Users\yalem\Documents\image reconstruction simulation testing'
os.chdir(path)

raw_frames = np.load('testing_frames_2.npy')
#print(raw_frames[1,:,:])
tp.quiet()
for i in range(raw_frames.shape[0]):
    if i ==0:
        f = tp.locate(raw_frames[i,:,:]*256, 25, invert=True, minmass=2000)
    else:
        f_new = tp.locate(raw_frames[i,:,:]*256, 25, invert=True, minmass=2000)
        f = pd.concat([f,f_new])


#print(f)
# fig1, ax00 = plt.subplots()
# ax00.hist(f['mass'], bins=1000)
# tp.subpx_bias(f)
# plt.show()


ypx = np.array(f.loc[:,'y'])
xpx = np.array(f.loc[:,'x'])
frames = np.arange(f.shape[0])
conv = 0.566 *1e-6
y = -1*(ypx - ypx[0])
x = (xpx - xpx[0])
fig2, ax2 = plt.subplots()
ax2.plot(frames[:100],y[:100],label='Y')
ax2.plot(frames[:100],x[:100], label='X')
handles, labels = ax2.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax2.legend(by_label.values(), by_label.keys(), fontsize=12, loc="lower right", borderaxespad=1)
ax2.grid()
plt.show()


positions = np.vstack([x,y])
print(positions)


np.save('trackpy_reconstruction.npy', positions)
