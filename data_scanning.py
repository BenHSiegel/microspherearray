# -*- coding: utf-8 -*-
"""
Created on Wed May  1 16:41:20 2024

@author: Ben
"""

import os
from folderscanning_videoprocessor import *


main_directory = r"C:\Users\bensi\Documents\Research\20240604"
pixtoum = 10/10  #diameter of sphere (um) / number of pixels for diameter of sphere
centroid_diameter = 13 #centroid size to look for in the images in pixels, always odd and overestimating size
pcacheck = False
saveposdata = True
saveFFTavg = True
        
for path, folders, files in os.walk(main_directory):
    for folder_name in folders:
        directory = f"{path}/{folder_name}"
        os.chdir(directory)
        print(folder_name)
        with open('info.txt') as file:
            lines = [line.rstrip() for line in file]
        framerate = float(lines[0])
        
        fftsave = f"{folder_name}_rmsavg"

        totalspheres = videofolder_dataextractions(directory, framerate, centroid_diameter, pixtoum, pcacheck, saveposdata)
        hdf5file_RMSprocessing(directory, totalspheres, saveFFTavg, fftsave)
        
    break
