from ipyparallel import Client
client = Client()
view = client.load_balanced_view()

%%px
import pims
import trackpy as tp
tp.quiet()
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