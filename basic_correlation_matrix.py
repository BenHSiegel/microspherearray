import os
import h5py
import numpy as np
import scipy
import matplotlib as mpl 
import matplotlib.pyplot as plt
from scipy.optimize import minimize, curve_fit
from matplotlib.pyplot import gca
import pandas as pd
from matplotlib.colors import LogNorm
import seaborn as sn
from correlationcalculator import heatmap
from correlationcalculator import annotate_heatmap

files = [r'F:\Lab data\20240604\0-8MHz\0.8correlationmatrix.h5', r'F:\Lab data\20240604\1MHz\1.0correlationmatrix.h5', r'F:\Lab data\20240604\1-25MHz\1.25correlationmatrix.h5', r'F:\Lab data\20240604\1-5MHz\1.5correlationmatrix.h5']
separations = ['56', '70', '88', '105']
mpl.rcParams.update({'font.size': 18})

fig, ax = plt.subplots(1, len(files))
cax = fig.add_axes(rect=(0.2,0.2,0.6,0.03))
#fig.suptitle('Correlation of Motion of a 5x5 Array of Spheres')
for i in range(len(files)):
    hf = h5py.File(files[i], 'r')

    totalspheres = 25
    xcor = hf.get('X_correlations')
    ycor = hf.get('Y_correlations')
    xcor = xcor[()]
    ycor = ycor[()]
    hf.close()

    for l in range(xcor.shape[0]):
        for m in range(xcor.shape[1]):

            if xcor[l][m] == 1:
                xcor[l][m] = 0

    print(np.max(xcor))
    print(np.min(xcor))


    spherenames = [str(x+1) for x in range(totalspheres)]
    #norm = LogNorm()
    if i == len(files) - 1:
        plot_cbar = True
        cbar_kws = {'shrink' : 0.8,
                    'orientation': 'horizontal'}
        cbar_ax = cax
        
    else:
        plot_cbar = False
        cbar_kws = None
        cbar_ax = None
    
    symcor = xcor
    for a in range(xcor.shape[0]):
        for b in range(xcor.shape[1]):
            if a < b:
                symcor[a][b] = ycor[a][b]
    mask = np.triu(np.ones_like(xcor, dtype=bool))
    diagmask = np.identity(xcor.shape[0])
    sn.heatmap(symcor, mask=diagmask, square=True, cmap = 'viridis', vmin=-0.25, vmax=0.25, ax=ax[i], cbar=plot_cbar, cbar_ax = cbar_ax, cbar_kws=cbar_kws)
    ax[i].tick_params(axis='both', which='major', labelsize=18)
    #ax[i].set_xticks(np.arange(xcor.shape[1])+.5, labels=spherenames,fontsize=16)
    #ax[i].set_yticks(np.arange(xcor.shape[0])+.5, labels=spherenames,fontsize=16)
    plt.setp(ax[i].get_xticklabels(), rotation=0)
    ax[i].set_title(separations[i] + r'$\mu$m Spacing', fontsize=26, pad=15)
    ax[i].set_xlabel('Sphere Index', fontsize=22, labelpad=5)
    ax[i].set_ylabel('Sphere Index',fontsize=22,labelpad=5)
fig.tight_layout()
cax.set_title('Correlation Coefficients',fontsize=26)
plt.show()
