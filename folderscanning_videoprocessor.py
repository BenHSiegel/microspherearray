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
import h5py
#import cv2 as cv

#make a pipeline so that when pims opens a file, it converts each frame to one color
@pims.pipeline
def gray(image):
    return image[:, :, 1]  # Take just the green channel (they are all the same for our camera)


def processmovie(filename, framerate, diameter):
    #open a avi file with pims and converts to one color
    spheres = gray(pims.open(filename))
    tp.quiet()
    #process every frame in the tiff image stack to find the locations of bright spots
    #minmass defines the minimum brightness and processes means no parallelization since that breaks it
    #invert=true looks for dark spots instead of light spots
    #diameter is the centroid size to look for in the images (in units of pixels)
    #diameter should always be an odd number and greater than the actual sphere size
    f = tp.batch(spheres[:], diameter, invert=True, minmass=350, processes=1)
        #to check the mass brightness make this figure
    # fig, ax = plt.subplots()
    # ax.hist(f['mass'], bins=1000)
    return [spheres, f]
    



def motiontracer(spheres, f):
    
    #look at the location in each frame and labels them. It looks for maximum 20 pixel movement between frames
    #if it vanishes for one frame, memory prevents it from thinking the sphere is gone (up to 10 frames)
    
    #suppress output so that it runs faster
    tp.quiet()

    t = tp.link(f, 10, memory=50)
    
    # fig1, ax00 = plt.subplots()
    # fig1.set_dpi(1200)

    #plot the trajectory of the sphere over the video
    # pixtoum = 10/11
    # tp.plot_traj(t, ax=ax00, label=False, mpp = pixtoum)
    
    # ax00.set_xlabel(r'x [$ \mu m$]')
    # ax00.set_ylabel(r'y [$ \mu m$]')
    # ax00.set_title("Spheres' Traces")
    return t

def lorentzian(f, f0, gam, cal_fac):
  kb = 1.38e-23 # Boltzmann's constant, SI units
  temp = 293 # Room temp, K
  m = 1e-12 # Mass kg
  omega = 2*np.pi*f
  omega0 = 2*np.pi*f0
  return 1/(cal_fac)**2 * 2*kb*temp/m * gam/((omega0**2 - omega**2)**2 + omega**2*gam**2)

def psdplotter(t, framerate, spheres, f, pixtoum, pcacheck, saveposdata, savename):
    ypx = t.loc[:,'y']
    xpx = t.loc[:,'x']
    spherenumber = t.loc[:,'particle']
    framenum = t.loc[:,'frame']
    #pixtoum = 10/13     #diameter of sphere (um) / number of pixels for diameter of sphere
    ypos = ypx * pixtoum * 10**(-6) #convert pixel to meter  (pixel dimension 4.8x4.8um)
    xpos = xpx * pixtoum * 10**(-6) #convert pixel to meter

    totalspheres = max(t.loc[:,'particle']) + 1 
    xposlist = [[] for i in range(totalspheres)]
    yposlist = [[] for i in range(totalspheres)]
    framenumlist = [[] for i in range(totalspheres)]

    #sort the dataframe to get the x,y position for each sphere
    for i in range(0, t.shape[0]):
        xposlist[spherenumber[i]].append(xpos[i])
        yposlist[spherenumber[i]].append(ypos[i])
        framenumlist[spherenumber[i]].append(framenum[i])
    
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
    sphere_pos_data = [[] for i in range(totalspheres)]
    '''
    Legendx = []
    Legendy = []
    figa, axa = plt.subplots()
    
    figa.set_size_inches(7.6, 4.5)
    figa.set_dpi(1200)
    
    figb, axb = plt.subplots()
    figb.set_size_inches(7.6, 4.5)
    figb.set_dpi(1200)
    '''
    fftbinning = 2048
    figs={}
    axs={}
    for i in range(0,len(xposlist)):
        
        if len(xposlist[i]) < nodrops:
            print('Sphere ' + str(i) + ' drops frames')
        else:
            frames = framenumlist[i]
            xcentered = xposlist[i] - np.mean(xposlist[i])
            xfreq, xPSD = welch(xcentered, framerate, 'hann', segmentsize, segmentsize/2, fftbinning, 'constant', True, 'density', 0,'mean')
            xASD = np.sqrt(xPSD)
            xASDlist[i] = np.vstack((xfreq,xASD)).T
            
            #axa.semilogy(xfreq, xASD, linewidth=2)
            #Legendx.append('Sphere ' + str(i))
    
            ycentered = yposlist[i] - np.mean(yposlist[i])
            yfreq, yPSD = welch(ycentered, framerate, 'hann', segmentsize, segmentsize/2, fftbinning, 'constant', True, 'density', 0,'mean')
            yASD = np.sqrt(yPSD)
            yASDlist[i] = np.vstack((yfreq,yASD)).T
            
            #axb.semilogy(yfreq, yASD, linewidth=2)
            #Legendy.append('Sphere ' + str(i))
            
            spheredata[i] = np.vstack((xcentered, ycentered)).T
            sphere_pos_data[i] = np.vstack((frames, xcentered, ycentered)).T
            
            if pcacheck == True:
                pca = PCA(n_components=2) ## keep 3 components (x,y,z)
                pca.fit(spheredata[i]) ## fit our data
                orig = pca.transform(spheredata[i]) ## reconstruct the original data from the PCA transform
                
                figs[i], axs[i] = plt.subplots(1, 2, sharey=False, tight_layout=True)
                
                figs[i].set_size_inches(18.5, 10.5)
                figs[i].set_dpi(800)
                plt.rcParams.update({'font.size': 22})
                
                xfreq_uncor, xPSD_uncor = welch(orig[:,0], framerate, 'hann', segmentsize, segmentsize/4, fftbinning, 'constant', True, 'density', 0,'mean')
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
                
                yfreq_uncor, yPSD_uncor = welch(orig[:,1], framerate, 'hann', segmentsize, segmentsize/4, fftbinning, 'constant', True, 'density', 0,'mean')
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

    '''
    axa.grid()
    axa.set_xlabel('Frequency [Hz]', fontsize=18)
    axa.set_ylabel(r'ASD [$m/ \sqrt{Hz}$]', fontsize=18)
    axa.legend(Legendx, fontsize=12, bbox_to_anchor=(1.04, 0), loc="lower left", borderaxespad=0)
    axa.set_title('X motion ASD', fontsize=22)
    axa.tick_params(axis='both', which='major', labelsize=12)
    #axa.set_xlim(8,167)
    for location in ['left', 'right', 'top', 'bottom']:
        axa.spines[location].set_linewidth(1)

    axb.grid()
    axb.set_xlabel('Frequency [Hz]', fontsize=18)
    axb.set_ylabel(r'ASD [$m/ \sqrt{Hz}$]', fontsize=18)
    axb.legend(Legendy, fontsize=12, bbox_to_anchor=(1.04, 0), loc="lower left", borderaxespad=0)
    axb.set_title('Y motion ASD', fontsize=22)
    axb.tick_params(axis='both', which='major', labelsize=12)
    #axb.set_xlim(8,167)
    for location in ['left', 'right', 'top', 'bottom']:
        axb.spines[location].set_linewidth(1)
    '''
    
    if saveposdata:
        savename = savename + '.h5'
        hf = h5py.File(savename, 'w')
        g1 = hf.create_group('position')
        g1.attrs.create('framerate (fps)', framerate)
        g2 = hf.create_group('X_psd')
        g2.attrs.create('framerate (fps)', framerate)
        g2.attrs.create('FFT bins', fftbinning)
        g2.attrs.create('FFT segment size', segmentsize)
        g3 = hf.create_group('Y_psd')
        g3.attrs.create('framerate (fps)', framerate)
        g3.attrs.create('FFT bins', fftbinning)
        g3.attrs.create('FFT segment size', segmentsize)
        for sphnum in range(len(spheredata)):
            d1 = g1.create_dataset('Sphere ' + str(sphnum), data=sphere_pos_data[sphnum])
            d1.attrs.create('range (m)', [np.ptp(sphere_pos_data[sphnum][:,1]), np.ptp(sphere_pos_data[sphnum][:,2])])
            d1.attrs.create('rms (m)', [np.sqrt(np.mean((sphere_pos_data[sphnum][:,1])**2)), np.sqrt(np.mean((sphere_pos_data[sphnum][:,2])**2))])
            g2.create_dataset('Sphere ' + str(sphnum), data=xASDlist[sphnum])
            g3.create_dataset('Sphere ' + str(sphnum), data=yASDlist[sphnum])
        hf.close()


    return totalspheres
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





# def average_size_calculator(filename):
    
#     img = 
#     blurredimg = cv.GaussianBlur(img, (3,3), 0)
#     circles = cv.HoughCircles(blurredimg, cv.HOUGH_GRADIENT, 1, 25, param1=50, param2=30, minRadius=5, maxRadius=30)
#     for i in circles[0,:]:
#      # draw the outer circle
#      cv.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
#      # draw the center of the circle
#      cv.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
     
#     cv.imshow('detected circles',cimg)
#     cv.waitKey(0)
#     cv.destroyAllWindows()

def videofolder_dataextractions(path, framerate, diameter, pixtoum, pcacheck, saveposdata):
    file_name_directory = []
    for filename in sorted(os.listdir(path)):
        if filename.endswith(".avi"):
            file_name_directory.append(filename)
         
    for vid in file_name_directory:
        [spheres, f] = processmovie(vid, framerate, diameter)
        t = motiontracer(spheres, f)
        totalspheres = psdplotter(t, framerate, spheres, f, pixtoum, pcacheck, saveposdata, vid[:-4])

    return totalspheres

def hdf5file_RMSprocessing(path, totalspheres, saveflag, savename):
    hdf5_list = []
    for filename in sorted(os.listdir(path)):
        if filename.endswith(".h5"):
            hdf5_list.append(filename)
    
    xfftmatrix = [[] for i in range(totalspheres)]
    yfftmatrix = [[] for i in range(totalspheres)]
    counter = 0
    for i in hdf5_list:
        hf = h5py.File(i, 'r')
        xgroup = hf.get('X_psd')
        ygroup = hf.get('Y_psd')
        
        k=0
        for j in xgroup.items():
            xfftj = np.array(j[1])
            if counter == 0:
                xfftmatrix[k] = xfftj[:,1].reshape(-1,1)
            else:
                xfftmatrix[k] = np.concatenate((xfftmatrix[k], xfftj[:,1].reshape(-1,1)), axis=1)
            k+=1
            
        k=0
        for j in ygroup.items():
            yfftj = np.array(j[1])
            if counter == 0:
                yfftmatrix[k] = yfftj[:,1].reshape(-1,1)
            else:
                yfftmatrix[k] = np.concatenate((yfftmatrix[k], yfftj[:,1].reshape(-1,1)), axis=1)
            freqs = yfftj[:,0]
            k+=1
        
        counter += 1
        hf.close()
            
    xrms_averaged = [[] for i in range(totalspheres)]
    yrms_averaged = [[] for i in range(totalspheres)]
    
    Legend = []
    
    figc, axc = plt.subplots()
    figc.set_size_inches(7.6, 4.5)
    figc.set_dpi(600)
    
    figd, axd = plt.subplots()
    figd.set_size_inches(7.6, 4.5)
    figd.set_dpi(600)
    for i in range(totalspheres):
        
        xrms_avg_i = np.sqrt(np.mean(xfftmatrix[i]**2, axis=1))
        yrms_avg_i = np.sqrt(np.mean(yfftmatrix[i]**2, axis=1))
        
        xrms_averaged[i] = xrms_avg_i
        yrms_averaged[i] = yrms_avg_i
        
        axc.semilogy(freqs, xrms_avg_i, linewidth=2)
        axd.semilogy(freqs, yrms_avg_i, linewidth=2)
        Legend.append('Sphere ' + str(i))
    
    axc.grid()
    #axc.set_xlim(5,180)
    axc.set_xlabel('Frequency [Hz]', fontsize=18)
    axc.set_ylabel(r'ASD [$m/ \sqrt{Hz}$]', fontsize=18)
    axc.legend(Legend, fontsize=12, bbox_to_anchor=(1.04, 0), loc="lower left", borderaxespad=0)
    axc.set_title('X motion RMS Avg ASD', fontsize=22)
    
    axd.grid()
    #axd.set_xlim(5,180)
    axd.set_xlabel('Frequency [Hz]', fontsize=18)
    axd.set_ylabel(r'ASD [$m/ \sqrt{Hz}$]', fontsize=18)
    axd.legend(Legend, fontsize=12, bbox_to_anchor=(1.04, 0), loc="lower left", borderaxespad=0)
    axd.set_title('Y motion RMS Avg ASD', fontsize=22)
    
    if saveflag:
        savename = savename + '.h5'
        hf = h5py.File(savename, 'w')
        hf.create_dataset('frequencies', data=freqs)
        hf.create_dataset('XASD RMS Avg', data=xrms_averaged)
        hf.create_dataset('YASD RMS Avg', data=yrms_averaged)

        hf.close()
        
# path = r"C:\Users\bensi\Documents\Research\20240424\charge check"
# os.chdir(path)
# framerate = 672
# pcacheck = False
# saveposdata = True
# #saveFFTavg = True
# #fftsave = "expandedposition20240319rmsavg"

# totalspheres = videofolder_dataextractions(path, framerate, pcacheck, saveposdata)
# #hdf5file_RMSprocessing(path, totalspheres, saveFFTavg, fftsave)
