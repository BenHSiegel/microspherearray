# New camera trial analysis


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
    return image[:, :,1]  # Take just the green channel (they are all the same for our camera)


def processmovie(filename, framerate, diameter):
    #open a avi file with pims and converts to one color
    spheres = gray(pims.open(filename))
    #plt.imshow(spheres[32]);
    tp.quiet()
    #process every frame in the tiff image stack to find the locations of bright spots
    #minmass defines the minimum brightness and processes means no parallelization since that breaks it
    #invert=true looks for dark spots instead of light spots
    #diameter is the centroid size to look for in the images (in units of pixels)
    #diameter should always be an odd number and greater than the actual sphere size
    f = tp.batch(spheres[:], diameter, invert=False, minmass=25000, processes=1)
        #to check the mass brightness make this figure
    fighist, axhist = plt.subplots()
    axhist.hist(f['mass'], bins=1000)
    plt.show()
    return [spheres, f]
    



def motiontracer(spheres, f):
    
    #look at the location in each frame and labels them. It looks for maximum 20 pixel movement between frames
    #if it vanishes for one frame, memory prevents it from thinking the sphere is gone (up to 10 frames)
    
    #suppress output so that it runs faster
    tp.quiet()

    t = tp.link(f, 50, memory=20)
    
    fig1, ax00 = plt.subplots()

    #plot the trajectory of the sphere over the video
    pixtoum = 0.566
    tp.plot_traj(t, ax=ax00, label=False, mpp = pixtoum)
    
    ax00.set_xlabel(r'x [$ \mu m$]')
    ax00.set_ylabel(r'y [$ \mu m$]')
    ax00.set_title("Spheres' Traces")
    plt.show()
    return t


def psdplotter(t, framerate, spheres, f, rowlen, pixtoum, pcacheck, saveposdata, savename, sortlabels = True):
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
    print(nodrops)
    print(totalspheres)
    if totalspheres > 1 and sortlabels == True:
        #sort the spheres by their position in the frame so it can be consistent across videos
        xposmeans = [np.average(xposlist[i]) for i in range(len(xposlist))]
        yposmeans = [np.average(yposlist[i]) for i in range(len(yposlist))]

        xsortedind = np.argsort(xposmeans)
        xsorted = [ xposmeans[i] for i in xsortedind ]
        ysorted = [ yposmeans[i] for i in xsortedind ]

        ysortedind = np.empty(0).astype(int)
        i = rowlen
        lasti = 0

        while i <= totalspheres:
            rowsort = np.argsort(ysorted[lasti:i])+lasti
            rowsort = rowsort.astype(int)
            ysortedind = np.concatenate((ysortedind, rowsort))
            lasti = i
            i = i + rowlen

        xposlist = [ xposlist[j] for j in xsortedind ]
        xposlist = [ xposlist[j] for j in ysortedind ]
        xmeanssorted = [ xsorted[j] for j in ysortedind]

        yposlist = [ yposlist[j] for j in xsortedind ]
        yposlist = [ yposlist[j] for j in ysortedind ]
        ymeanssorted = [ ysorted[j] for j in ysortedind ]
    else:
        xmeanssorted = [np.average(xposlist[i]) for i in range(len(xposlist))]
        ymeanssorted = [np.average(yposlist[i]) for i in range(len(yposlist))]

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

    Legendx = []
    Legendy = []
    figa, axa = plt.subplots()

    figb, axb = plt.subplots()

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
            
            axa.semilogy(xfreq, xASD, linewidth=2)
            Legendx.append('Sphere ' + str(i))
    
            ycentered = yposlist[i] - np.mean(yposlist[i])
            yfreq, yPSD = welch(ycentered, framerate, 'hann', segmentsize, segmentsize/2, fftbinning, 'constant', True, 'density', 0,'mean')
            yASD = np.sqrt(yPSD)
            yASDlist[i] = np.vstack((yfreq,yASD)).T
            
            axb.semilogy(yfreq, yASD, linewidth=2)
            Legendy.append('Sphere ' + str(i))
            
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
            d1.attrs.create('Camera frame location (m)', [xmeanssorted[sphnum], ymeanssorted[sphnum]])
            d1.attrs.create('Camera pixel location', [np.mean(xpx),np.mean(ypx)])
            g2.create_dataset('Sphere ' + str(sphnum), data=xASDlist[sphnum])
            g3.create_dataset('Sphere ' + str(sphnum), data=yASDlist[sphnum])
        hf.close()

    plt.show()
    return totalspheres


path = r"C:\Users\Ben\Documents\Research\20240917\beforechamber"
os.chdir(path)
vid='modbeams.avi'
diameter = 85
pixtoum = 10
framerate = 1500
pcacheck = False
saveposdata = True
saveFFTavg = False
rowlen = 3
fftsave = "chargecheck"

[spheres, f] = processmovie(vid, framerate, diameter)
t = motiontracer(spheres, f)
totalspheres = psdplotter(t, framerate, spheres, f, rowlen, pixtoum, pcacheck, saveposdata, vid[:-4], False)