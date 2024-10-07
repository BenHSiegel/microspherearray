
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
    f = tp.batch(spheres[:1000], diameter, invert=True, minmass=400, processes=1)
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
    
    fig1, ax00 = plt.subplots()
    fig1.set_size_inches(6,6)
    #plot the trajectory of the sphere over the video
    pixtoum = 10/8
    tp.plot_traj(t, ax=ax00, label=False, superimpose=spheres[100])
    
    ax00.set_xlabel(r'X (px)',fontsize = 18)
    ax00.set_ylabel(r'Y (px)',fontsize = 18)
    ax00.set_title("Traces of Motion from Video",fontsize = 22)
    ax00.tick_params(labelsize=14)
    plt.savefig('trace_1.svg',format='svg', dpi=600)
    plt.show()
    return t

path = r'C:\Users\Ben\Documents\Research\Moore Lab\figures'
os.chdir(path)
vid = r'E:\Lab data\20240604\0-8MHz\0-8MHz_25grid-lp-2.avi'
framerate = 1000.25
diameter=15
[spheres, f] = processmovie(vid, framerate, diameter)
t = motiontracer(spheres, f)

