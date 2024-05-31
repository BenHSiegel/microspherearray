# -*- coding: utf-8 -*-
"""
Created on Wed May  1 16:41:20 2024

@author: Ben
"""

import os
from folderscanning_videoprocessor import *


main_directory = r"D:\Lab data\20240529"

for path, folders, files in os.walk(main_directory):
    for folder_name in folders:
        directory = f"{path}/{folder_name}"
        os.chdir(directory)
        print(folder_name)
        with open('info.txt') as file:
            lines = [line.rstrip() for line in file]
        framerate = float(lines[0])
        pcacheck = False
        saveposdata = True
        saveFFTavg = True
        fftsave = f"{folder_name}_rmsavg"

        totalspheres = videofolder_dataextractions(directory, framerate, pcacheck, saveposdata)
        hdf5file_RMSprocessing(directory, totalspheres, saveFFTavg, fftsave)
        
    break
