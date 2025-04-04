# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 10:27:38 2023

@author: Ben
"""

import av as av
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
from scipy.signal import welch
from scipy.signal.windows import blackman
from scipy.signal import find_peaks
from matplotlib.pyplot import gca


#make a pipeline so that when pims opens a file, it converts each frame to one color
@pims.pipeline
def gray(image):
    return image[:, :, 1]  # Take just the green channel (they are all the same for our camera)


def processmovie(filename, framerate):
    #open a avi file with pims and converts to one color
    spheres = gray(pims.open(filename))
    tp.quiet()
    #process every frame in the tiff image stack to find the locations of bright spots
    #minmass defines the minimum brightness and processes means no parallelization since that breaks it
    #invert=true looks for dark spots instead of light spots
    f = tp.batch(spheres[:500], 13, invert=True, minmass=400, processes='auto')
    #to check the mass brightness make this figure
    fig, ax = plt.subplots()

    ax.hist(f['mass'], bins=100)

    return [spheres, f]
    



def motiontracer(spheres, f):
    
    #look at the location in each frame and labels them. It looks for maximum 5 pixel movement between frames
    #if it vanishes for one frame, memory prevents it from thinking the sphere is gone (up to 3 frames)
    
    #suppress output so that it runs faster
    tp.quiet()

    t = tp.link(f, 20, memory=10)
    
    fig1, ax00 = plt.subplots()
    fig1.set_dpi(1200)

    #plot the trajectory of the sphere over the video
    pixtoum = 10/11
    tp.plot_traj(t, ax=ax00, label=False, mpp = pixtoum)
    
    ax00.set_xlabel(r'x [$ \mu m$]')
    ax00.set_ylabel(r'y [$ \mu m$]')
    ax00.set_title("Spheres' Traces")
    return t

def lorentzian(f, f0, gam, cal_fac):
  kb = 1.38e-23 # Boltzmann's constant, SI units
  temp = 293 # Room temp, K
  m = 1e-12 # Mass kg
  omega = 2*np.pi*f
  omega0 = 2*np.pi*f0
  return 1/(cal_fac)**2 * 2*kb*temp/m * gam/((omega0**2 - omega**2)**2 + omega**2*gam**2)

def psdplotter(t,framerate,spheres,f):
    ypx = t.loc[:,'y']
    xpx = t.loc[:,'x']
    spherenumber = t.loc[:,'particle']
    pixtoum = 10/11
    ypos = ypx * pixtoum * 10**(-6) #convert pixel to meter  (pixel dimension 4.8x4.8um)
    xpos = xpx * pixtoum * 10**(-6) #convert pixel to meter

    totalspheres = max(t.loc[:,'particle']) + 1 
    xposlist = [[] for i in range(totalspheres)]
    yposlist = [[] for i in range(totalspheres)]


    #sort the dataframe to get the x,y position for each sphere
    for i in range(0, t.shape[0]):
        xposlist[spherenumber[i]].append(xpos[i])
        yposlist[spherenumber[i]].append(ypos[i])
    
    nodrops = max(len(i) for i in xposlist)

    #make an array of the time for each frame in the video
    timeinc = 1/framerate 
    numframes = len(spheres) #gets number of frames in the video
    time = np.arange(0, numframes*timeinc, timeinc)
    #freq = fft.rfftfreq(numframes, timeinc)
    #w = blackman(numframes)
    segmentsize = round(framerate/4)

    xASDlist = [[] for i in range(totalspheres)]
    yASDlist = [[] for i in range(totalspheres)]

    spheredata = [[] for i in range(totalspheres)]
    
    Legendx = []
    Legendy = []
    figa, axa = plt.subplots()
    
    figa.set_size_inches(7.6, 4.5)
    figa.set_dpi(1200)
    
    figb, axb = plt.subplots()
    figb.set_size_inches(7.6, 4.5)
    figb.set_dpi(1200)
    figs={}
    axs={}
    for i in range(0,len(xposlist)):
        
        if len(xposlist[i]) < nodrops:
            print('Sphere ' + str(i) + ' drops frames')
        else:
            xcentered = xposlist[i] - np.mean(xposlist[i])
            xfreq, xPSD = welch(xcentered, framerate, 'hann', segmentsize, segmentsize/2, None, 'constant', True, 'density', 0,'mean')
            #xPSD = 2 * timeinc / numframes * np.abs(fft.rfft(xcentered))**2
            xASDlist[i] = np.sqrt(xPSD)
            
            axa.loglog(xfreq, xASDlist[i], linewidth=2)
            Legendx.append('Sphere ' + str(i))
    
            ycentered = yposlist[i] - np.mean(yposlist[i])
            yfreq, yPSD = welch(ycentered, framerate, 'hann', segmentsize, segmentsize/2, None, 'constant', True, 'density', 0,'mean')
            #yPSD = 2 * timeinc / numframes * np.abs(fft.rfft(ycentered))**2
            yASDlist[i] = np.sqrt(yPSD)
            
            axb.loglog(yfreq, yASDlist[i], linewidth=2)
            Legendy.append('Sphere ' + str(i))
            
            spheredata[i] = np.vstack((xcentered, ycentered)).T
            
            pca = PCA(n_components=2) ## keep 3 components (x,y,z)
            pca.fit(spheredata[i]) ## fit our data
            orig = pca.transform(spheredata[i]) ## reconstruct the original data from the PCA transform
            
            figs[i], axs[i] = plt.subplots(1, 2, sharey=False, tight_layout=True)
            
            figs[i].set_size_inches(18.5, 10.5)
            figs[i].set_dpi(1200)
            plt.rcParams.update({'font.size': 22})
            
            xfreq_uncor, xPSD_uncor = welch(orig[:,0], framerate, 'hann', segmentsize, segmentsize/4, None, 'constant', True, 'density', 0,'mean')
            init_guessx = [xfreq_uncor[np.argmax(xPSD_uncor)],70,1e-7] # guess for the initial parameters
            best_paramsx, cov = curve_fit(lorentzian, xfreq_uncor, xPSD_uncor, p0=init_guessx)
            
            
            axs[i][0].loglog(xfreq_uncor, xPSD_uncor, 'k', label = "Data")
            axs[i][0].plot(xfreq_uncor, lorentzian(xfreq_uncor, *best_paramsx), 'r', label="Fit")
            
            axs[i][0].set_ylim([1E-17,None])

            peaks1, _ = find_peaks(xPSD_uncor,threshold=1E-15)
            for j, txt in enumerate(np.around(xfreq_uncor[peaks1])):
                axs[i][0].annotate(txt, (xfreq_uncor[peaks1[j]],xPSD_uncor[peaks1[j]]))
            
            figs[i].suptitle("Sphere " + str(i) + ' uncorrelation attempt')
            
            axs[i][0].set_xlabel('Frequency [Hz]')
            axs[i][0].set_ylabel(r'PSD [$m^2/ Hz$]')
            axs[i][0].set_title('X PSD')
            axs[i][0].legend()
            
            yfreq_uncor, yPSD_uncor = welch(orig[:,1], framerate, 'hann', segmentsize, segmentsize/4, None, 'constant', True, 'density', 0,'mean')
            init_guessy = [yfreq_uncor[np.argmax(yPSD_uncor)],70,1E-7] # guess for the initial parameters
            best_paramsy, cov = curve_fit(lorentzian, yfreq_uncor, yPSD_uncor, p0=init_guessy)
            
            
            axs[i][1].loglog(yfreq_uncor, yPSD_uncor, 'k', label = "Data")
            axs[i][1].plot(yfreq_uncor, lorentzian(yfreq_uncor, *best_paramsy), 'r', label="Fit")
            
            peaks2, _ = find_peaks(yPSD_uncor,threshold=1E-15)
            for j, txt in enumerate(np.around(yfreq_uncor[peaks2])):
                axs[i][1].annotate(txt, (yfreq_uncor[peaks2[j]],yPSD_uncor[peaks2[j]]))
            
            axs[i][1].set_ylim([1E-17,None])
            
            axs[i][1].set_xlabel('Frequency [Hz]')
            axs[i][1].set_ylabel(r'PSD [$m^2/ Hz$]')
            axs[i][1].set_title('Y PSD')

    print(max(yfreq))
    axa.grid()
    axa.set_xlabel('Frequency [Hz]', fontsize=18)
    axa.set_ylabel(r'ASD [$m/ \sqrt{Hz}$]', fontsize=18)
    axa.legend(Legendx, loc="lower left", fontsize=22)
    axa.set_title('X motion ASD', fontsize=22)
    axa.tick_params(axis='both', which='major', labelsize=12)
    axa.set_xlim(8,167)
    for location in ['left', 'right', 'top', 'bottom']:
        axa.spines[location].set_linewidth(1)

    axb.grid()
    axb.set_xlabel('Frequency [Hz]', fontsize=18)
    axb.set_ylabel(r'ASD [$m/ \sqrt{Hz}$]', fontsize=18)
    axb.legend(Legendy, loc="upper left", fontsize=12)
    axb.set_title('Y motion ASD', fontsize=22)
    axb.tick_params(axis='both', which='major', labelsize=12)
    axb.set_xlim(8,167)
    for location in ['left', 'right', 'top', 'bottom']:
        axb.spines[location].set_linewidth(1)
    

    

    # plt.ylim([1e-8, 4e-6])
    # #plt.xlim([1,100])

    
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



path = r"C:\Users\bensi\Documents\Research\20240319"
os.chdir(path)
filename = 'middle0-6mbar_vid4.avi'
framerate = 1248
[spheres, f] = processmovie(filename, framerate)
t = motiontracer(spheres, f)
psdplotter(t, framerate, spheres, f)