# -*- coding: utf-8 -*-
"""
Created on Wed May  1 16:41:20 2024

@author: Ben
"""

import os
from folderscanning_videoprocessor import *


main_directory = r"C:\Users\Ben\Documents\Research\20240430"

for path, folders, files in os.walk(main_directory):
    for folder_name in folders:
        directory = f"{path}/{folder_name}"
        os.chdir(directory)
        
        f = open(f"{folder_name}info.txt", 'r')
        freq = f.read()
        f.close()
        framerate = float(freq)
        pcacheck = False
        saveposdata = True
        saveFFTavg = True
        fftsave = f"{folder_name}_rmsavg"

        totalspheres = videofolder_dataextractions(directory, framerate, pcacheck, saveposdata)
        #hdf5file_RMSprocessing(directory, totalspheres, saveFFTavg, fftsave)
        
    break
