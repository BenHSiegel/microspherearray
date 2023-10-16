# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 10:27:38 2023

@author: Ben
"""

import av as av
import collections.abc
collections.Iterable = collections.abc.Iterable
collections.Iterator = collections.abc.Iterator
import pims
import trackpy as tp
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import fft
from sklearn.decomposition import PCA ## need this for principle component analysis below
from matplotlib.mlab import psd
from scipy.optimize import minimize, curve_fit
from scipy.stats import chi2
import os
from scipy.signal.windows import blackman


#make a pipeline so that when pims opens a file, it converts each frame to one color
@pims.pipeline
def gray(image):
    return image[:, :, 1]  # Take just the green channel (they are all the same for our camera)


def processmovie(filename, framerate):
    #open a avi file with pims and converts to one color
    spheres = gray(pims.open(filename))

    #process every frame in the tiff image stack to find the locations of bright spots
    #minmass defines the minimum brightness and processes means no parallelization since that breaks it
    #invert=true looks for dark spots instead of light spots
    f = tp.batch(spheres[:], 31, invert=False, minmass=1000, processes=1)
    #to check the mass brightness make this figure
    fig, ax = plt.subplots()
    ax.hist(f['mass'], bins=100)

    return [spheres, f]
    



def PSDmaker(spheres, f):
    
    #look at the location in each frame and labels them. It looks for maximum 5 pixel movement between frames
    #if it vanishes for one frame, memory prevents it from thinking the sphere is gone (up to 3 frames)
    t = tp.link(f, 20, memory=10)
    #suppress output so that it runs faster
    tp.quiet()
    
    plt.figure(0)
    #plot the trajectory of the sphere over the video
    tp.plot_traj(t)

    ypx = t.loc[:,'y']
    xpx = t.loc[:,'x']
    spherenumber = t.loc[:,'particle']
    ypos = ypx * 4.8 * 10**(-6) #convert pixel to meter (pixel dimension 4.8x4.8um)
    xpos = xpx * 4.8 * 10**(-6) #convert pixel to meter

    totalspheres = max(t.loc[:,'particle']) + 1 
    xposlist = [[] for i in range(totalspheres)]
    yposlist = [[] for i in range(totalspheres)]


    #sort the dataframe to get the x,y position for each sphere
    for i in range(0, t.shape[0]):
        xposlist[spherenumber[i]].append(xpos[i])
        yposlist[spherenumber[i]].append(ypos[i])


    #make an array of the time for each frame in the video
    timeinc = 1/framerate 
    numframes = len(spheres) #gets number of frames in the video
    time = np.arange(0, numframes*timeinc, timeinc)
    freq = fft.rfftfreq(numframes, timeinc)
    w = blackman(numframes)


    xASDlist = [[] for i in range(totalspheres)]
    yASDlist = [[] for i in range(totalspheres)]

    for i in range(0,len(xposlist)):
        
        xcentered = xposlist[i] - np.mean(xposlist[i])
        xPSD = 2 * timeinc / numframes * np.abs(fft.rfft(xcentered*w))**2
        xASD = np.sqrt(xPSD)
        xASDlist[i] = xASD
        

        ycentered = yposlist[i] - np.mean(yposlist[i])
        yPSD = 2 * timeinc / numframes * np.abs(fft.rfft(ycentered*w))**2
        yASD = np.sqrt(yPSD)
        yASDlist[i] = yASD
        


    Legend = []
    plt.figure(1)
    for i in range(0,len(xASDlist)):
        if len(freq) == len(xASDlist[i]):
            plt.loglog(freq, xASDlist[i])
            Legend.append('Sphere ' + str(i+1))
        else:
            print('Sphere ' +str(i+1) + ' has dropped frames')
        
    #plt.ylim([1e-8, 2e-7])
    #plt.xlim([1,100])
    plt.xlabel('Frequency [Hz]')
    plt.ylabel(r'ASD [$m/ \sqrt{Hz}$]')
    #plt.legend(Legend)
    plt.title('X motion ASD')
    plt.grid()
    
    
    Legend = []
    plt.figure(2)
    for i in range(0,len(yASDlist)):
        if len(freq) == len(yASDlist[i]):
            plt.loglog(freq, yASDlist[i])
            Legend.append('Sphere ' + str(i+1))
        else:
            print('Sphere ' +str(i+1) + ' has dropped frames')
        
    #plt.ylim([1e-8, 2e-7])
    #plt.xlim([1,100])
    plt.xlabel('Frequency [Hz]')
    plt.ylabel(r'ASD [$m/ \sqrt{Hz}$]')
    #plt.legend(Legend)
    plt.title('Y motion ASD')
    plt.grid()
    
#     sphere2 = plt.figure(2)
#     plt.loglog(freq, x1ASD)
#     plt.loglog(freq, y1ASD)
#     plt.ylim([1e-9, 1e-3])
#     plt.xlabel('Frequency [Hz]')
#     plt.ylabel(r'ASD [$m/ \sqrt{Hz}$]')
#     plt.legend(['X motion', 'Y motion'])
#     plt.title('Sphere 2 Driven Motion')
    
    #sphere1.savefig('spheremovingpsd.png')
    #sphere2.savefig('sphere2movingpsd.png')
    #tracks.savefig('tracks.png')
    
#     rmsparsevalcheck0 = np.mean(x0centered**2)
#     psdparsevalcheck0 = 1/(numframes*timeinc) * np.sum(x0PSD)
#     print(rmsparsevalcheck0)
#     print(psdparsevalcheck0)
    
#     rmsparsevalcheck1 = np.mean(x1centered**2)
#     psdparsevalcheck1 = 1/(numframes*timeinc) * np.sum(x1PSD)
#     print(rmsparsevalcheck1)
#     print(psdparsevalcheck1)

    return t


path = r"C:\Users\Ben\Documents\Image processing jupyter notebooks"
os.chdir(path)
filename = 'beam jitter pre chamber.avi'
framerate = 304.23
[spheres, f] = processmovie(filename, framerate)

t = PSDmaker(spheres, f)