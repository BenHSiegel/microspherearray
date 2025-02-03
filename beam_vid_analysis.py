
import pims
import trackpy as tp
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import fft
from sklearn.decomposition import PCA ## need this for principle component analysis below
from matplotlib.mlab import psd
from scipy.optimize import minimize, curve_fit
from scipy.stats import chi2, linregress
import os
from scipy.signal import welch
from scipy.signal.windows import blackman
from scipy.signal import find_peaks
from matplotlib.pyplot import gca
import h5py


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
    f = tp.batch(spheres[:], 75, invert=False, minmass=20000, processes=1)
    #to check the mass brightness make this figure
    fig, ax = plt.subplots()
    ax.hist(f['mass'], bins=100)
    plt.show()
    return [spheres, f]
    

def singlebeamPSD(spheres,f,framerate,rowlen=1):
    tp.quiet()
    #look at the location in each frame and labels them. It looks for maximum 5 pixel movement between frames
    #if it vanishes for one frame, memory prevents it from thinking the sphere is gone (up to 3 frames)
    t = tp.link(f, 20, memory=10)
    #suppress output so that it runs faster
    
    tp.subpx_bias(tp.locate(spheres[0], 25, invert=False, minmass=10000))
    fig00, ax00 = plt.subplots()

    #plot the trajectory of the sphere over the video
    pixtoum = 4.8 #pixel is 4.8 um for old high speed camera
    tp.plot_traj(t, ax=ax00, label=False, mpp = pixtoum)
    
    ax00.set_xlabel(r'x [$ \mu m$]')
    ax00.set_ylabel(r'y [$ \mu m$]')
    ax00.set_title("Spheres' Traces")

    plt.show()

    ypx = t.loc[:,'y']
    xpx = t.loc[:,'x']
    spherenumber = t.loc[:,'particle']
    framenum = t.loc[:,'frame']
    ypos = ypx * 4.8 * 10**(-6) #convert pixel to meter (pixel dimension 4.8x4.8um)
    xpos = xpx * 4.8 * 10**(-6) #convert pixel to meter

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

    #sort the spheres by their position in the frame so it can be consistent across videos
    xposmeans = [np.average(xposlist[i]) for i in range(len(xposlist))]
    yposmeans = [np.average(yposlist[i]) for i in range(len(yposlist))]

    xsortedind = np.argsort(xposmeans)
    xsorted = [ xposmeans[i] for i in xsortedind ]
    ysorted = [ yposmeans[i] for i in xsortedind ]

    ysortedind = np.empty(0).astype(int)
    i = rowlen
    lasti = 0
    rowinds = []
    while i <= totalspheres:
        rowsort = np.argsort(ysorted[lasti:i])+lasti
        rowsort = rowsort.astype(int)
        ysortedind = np.concatenate((ysortedind, rowsort))
        rowinds.append([lasti,i])
        lasti = i
        i = i + rowlen

    xposlist = [ xposlist[j] for j in xsortedind ]
    xposlist = [ xposlist[j] for j in ysortedind ]
    xmeanssorted = [ xsorted[j] for j in ysortedind]

    yposlist = [ yposlist[j] for j in xsortedind ]
    yposlist = [ yposlist[j] for j in ysortedind ]
    ymeanssorted = [ ysorted[j] for j in ysortedind ]

    #make an array of the time for each frame in the video
    timeinc = 1/framerate 
    numframes = len(spheres) #gets number of frames in the video
    time = np.arange(0, numframes*timeinc, timeinc)
    freq = fft.rfftfreq(numframes, timeinc)
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


    axa.grid()
    axa.set_xlabel('Frequency (Hz)', fontsize=18)
    axa.set_ylabel(r'ASD ($m/ \sqrt{Hz}$)', fontsize=18)
    axa.legend(Legendx, fontsize=12, bbox_to_anchor=(1.04, 0), loc="lower left", borderaxespad=0)
    axa.set_title('X motion ASD', fontsize=22)
    axa.tick_params(axis='both', which='major', labelsize=12)
    #axa.set_xlim(8,167)
    for location in ['left', 'right', 'top', 'bottom']:
        axa.spines[location].set_linewidth(1)

    axb.grid()
    axb.set_xlabel('Frequency (Hz)', fontsize=18)
    axb.set_ylabel(r'ASD ($m/ \sqrt{Hz}$)', fontsize=18)
    axb.legend(Legendy, fontsize=12, bbox_to_anchor=(1.04, 0), loc="lower left", borderaxespad=0)
    axb.set_title('Y motion ASD', fontsize=22)
    axb.tick_params(axis='both', which='major', labelsize=12)
    #axb.set_xlim(8,167)
    for location in ['left', 'right', 'top', 'bottom']:
        axb.spines[location].set_linewidth(1)


def PSDmaker(spheres, f, framerate, rowlen, saveposdata, savename):
    tp.quiet()
    #look at the location in each frame and labels them. It looks for maximum 5 pixel movement between frames
    #if it vanishes for one frame, memory prevents it from thinking the sphere is gone (up to 3 frames)
    t = tp.link(f, 20, memory=10)
    #suppress output so that it runs faster
    
    tp.subpx_bias(tp.locate(spheres[0], 25, invert=False, minmass=10000))
    fig00, ax00 = plt.subplots()

    #plot the trajectory of the sphere over the video
    pixtoum = 4.8 #pixel is 4.8 um for old high speed camera
    tp.plot_traj(t, ax=ax00, label=False, mpp = pixtoum)
    
    ax00.set_xlabel(r'x [$ \mu m$]')
    ax00.set_ylabel(r'y [$ \mu m$]')
    ax00.set_title("Spheres' Traces")

    plt.show()


    ypx = t.loc[:,'y']
    xpx = t.loc[:,'x']
    spherenumber = t.loc[:,'particle']
    framenum = t.loc[:,'frame']
    ypos = ypx * 4.8 * 10**(-6) #convert pixel to meter (pixel dimension 4.8x4.8um)
    xpos = xpx * 4.8 * 10**(-6) #convert pixel to meter

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

    #sort the spheres by their position in the frame so it can be consistent across videos
    xposmeans = [np.average(xposlist[i]) for i in range(len(xposlist))]
    yposmeans = [np.average(yposlist[i]) for i in range(len(yposlist))]

    xsortedind = np.argsort(xposmeans)
    xsorted = [ xposmeans[i] for i in xsortedind ]
    ysorted = [ yposmeans[i] for i in xsortedind ]

    ysortedind = np.empty(0).astype(int)
    i = rowlen
    lasti = 0
    rowinds = []
    while i <= totalspheres:
        rowsort = np.argsort(ysorted[lasti:i])+lasti
        rowsort = rowsort.astype(int)
        ysortedind = np.concatenate((ysortedind, rowsort))
        rowinds.append([lasti,i])
        lasti = i
        i = i + rowlen

    xposlist = [ xposlist[j] for j in xsortedind ]
    xposlist = [ xposlist[j] for j in ysortedind ]
    xmeanssorted = [ xsorted[j] for j in ysortedind]

    yposlist = [ yposlist[j] for j in xsortedind ]
    yposlist = [ yposlist[j] for j in ysortedind ]
    ymeanssorted = [ ysorted[j] for j in ysortedind ]

    figx, axx = plt.subplots(1,2)
    figy, axy = plt.subplots(1,2)
    RFtones = [22,23,24,25,26]
    i = 0
    for k in rowinds:
        axx[0].plot(RFtones, xmeanssorted[k[0]:k[1]], '.')
        slope, intercept, r, p, se = linregress(RFtones, xmeanssorted[k[0]:k[1]])
        linechar = 'Row %d :m= %.2e -> %.2e' %(i, slope, slope/7)
        linefit = [ slope * n + intercept for n in RFtones]
        axx[0].plot(RFtones, linefit, 'r', linestyle = '--', alpha = 0.3, label = linechar)

        axx[1].plot(RFtones, ymeanssorted[k[0]:k[1]], '.')
        slope, intercept, r, p, se = linregress(RFtones, ymeanssorted[k[0]:k[1]])
        linechar = 'Row %d :m= %.2e -> %.2e' %(i, slope, slope/7)
        linefit = [ slope * n + intercept for n in RFtones]
        axx[1].plot(RFtones, linefit, 'r', linestyle = '--', alpha = 0.3, label = linechar)
        i += 1
    axx[0].legend()
    axx[1].legend()
    axx[0].set_xlabel("Driving Frequencies of Y Crystal (MHz)")
    axx[1].set_xlabel("Driving Frequencies of Y Crystal (MHz)")
    axx[0].set_ylabel("X Coordinate on Camera (m)")
    axx[1].set_ylabel("Y Coordinate on Camera (m)")
    figx.suptitle("Separation of Array vs Y Driving Frequencies")

    #only works for square arrays right now
    colind0 = np.arange(0,totalspheres, rowlen)
    for k in range(rowlen):
        colinds = colind0 + k
        print(len(xmeanssorted))
        print(xmeanssorted)
        colxdata = [xmeanssorted[n] for n in colinds]
        axy[0].plot(RFtones, colxdata, '.')
        slope, intercept, r, p, se = linregress(RFtones, colxdata)
        linechar = 'Column %d :m= %.2e -> %.2e' %(k, slope, slope/7)
        linefit = [ slope * n + intercept for n in RFtones]
        axy[0].plot(RFtones, linefit, 'r', linestyle = '--', alpha = 0.3, label = linechar)

        colydata = [ymeanssorted[n] for n in colinds]
        axy[1].plot(RFtones, colydata, '.')
        slope, intercept, r, p, se = linregress(RFtones, colydata)
        linechar = 'Column %d :m= %.2e -> %.2e' %(k, slope, slope/7)
        linefit = [ slope * n + intercept for n in RFtones]
        axy[1].plot(RFtones, linefit, 'r', linestyle = '--', alpha = 0.3, label = linechar)
    
    axy[0].legend()
    axy[1].legend()
    axy[0].set_xlabel("Driving Frequencies of X Crystal (MHz)")
    axy[1].set_xlabel("Driving Frequencies of X Crystal (MHz)")
    axy[0].set_ylabel("X Coordinate on Camera (m)")
    axy[1].set_ylabel("Y Coordinate on Camera (m)")
    figy.suptitle("Separation of Array vs X Driving Frequencies")


    #make an array of the time for each frame in the video
    timeinc = 1/framerate 
    numframes = len(spheres) #gets number of frames in the video
    time = np.arange(0, numframes*timeinc, timeinc)
    freq = fft.rfftfreq(numframes, timeinc)
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


    axa.grid()
    axa.set_xlabel('Frequency (Hz)', fontsize=18)
    axa.set_ylabel(r'ASD ($m/ \sqrt{Hz}$)', fontsize=18)
    axa.legend(Legendx, fontsize=12, bbox_to_anchor=(1.04, 0), loc="lower left", borderaxespad=0)
    axa.set_title('X motion ASD', fontsize=22)
    axa.tick_params(axis='both', which='major', labelsize=12)
    #axa.set_xlim(8,167)
    for location in ['left', 'right', 'top', 'bottom']:
        axa.spines[location].set_linewidth(1)

    axb.grid()
    axb.set_xlabel('Frequency (Hz)', fontsize=18)
    axb.set_ylabel(r'ASD ($m/ \sqrt{Hz}$)', fontsize=18)
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
            d1 = g1.create_dataset('Spot ' + str(sphnum), data=sphere_pos_data[sphnum])
            d1.attrs.create('range (m)', [np.ptp(sphere_pos_data[sphnum][:,1]), np.ptp(sphere_pos_data[sphnum][:,2])])
            d1.attrs.create('rms (m)', [np.sqrt(np.mean((sphere_pos_data[sphnum][:,1])**2)), np.sqrt(np.mean((sphere_pos_data[sphnum][:,2])**2))])
            d1.attrs.create('Camera frame location (m)', [xmeanssorted[sphnum], ymeanssorted[sphnum]])
            g2.create_dataset('Spot ' + str(sphnum), data=xASDlist[sphnum])
            g3.create_dataset('Spot ' + str(sphnum), data=yASDlist[sphnum])
        hf.close()

    plt.show()

    return

#     rmsparsevalcheck0 = np.mean(x0centered**2)
#     psdparsevalcheck0 = 1/(numframes*timeinc) * np.sum(x0PSD)
#     print(rmsparsevalcheck0)
#     print(psdparsevalcheck0)
    
#     rmsparsevalcheck1 = np.mean(x1centered**2)
#     psdparsevalcheck1 = 1/(numframes*timeinc) * np.sum(x1PSD)
#     print(rmsparsevalcheck1)
#     print(psdparsevalcheck1)


path = r"C:\Users\bensi\Documents\Research\20250203"
os.chdir(path)
framerate = 5000
saveposdata = False
rowlen = 1
savename = "aodjitter"
filename = 'farfromlens.avi'

[spheres, f] = processmovie(filename, framerate)

singlebeamPSD(spheres, f, framerate)