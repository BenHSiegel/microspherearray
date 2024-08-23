# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 13:41:11 2024

@author: Ben
"""
import numpy as np
import cv2 as cv
import os

path = r"D:\Lab data\20240604\0-8MHz"
filename = "pictry.png"
os.chdir(path)


    

img = cv.imread(filename)

assert img is not None, "file could not be read, check with os.path.exists()"
gimg = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
blurredimg = cv.GaussianBlur(gimg, (3,3), 0)
circles = cv.HoughCircles(blurredimg, cv.HOUGH_GRADIENT, 1, 25, param1=20, param2=5, minRadius=2, maxRadius=20)
circlespx = np.uint16(np.around(circles))
for i in circlespx[0,:]:
    # draw the outer circle
    cv.circle(img,(i[0],i[1]),i[2],(0,255,0),3)
    # draw the center of the circle
    cv.circle(img,(i[0],i[1]),2,(0,0,255),1)
    
    
cv.imshow('detected circles',img)
cv.waitKey(0)
cv.destroyAllWindows()
