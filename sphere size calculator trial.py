# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 13:41:11 2024

@author: Ben
"""
import numpy as np
import cv2 as cv
import os

path = r"C:\Users\Ben\Documents\Research\20240604\0-8MHz"
filename = r"0-8MHz_25grid-lp-1.avi"
os.chdir(path)

def average_size_calculator(filename):
    
    vid = cv.VideoCapture(filename)
    success, img = vid.read()
    if success:
        gimg = img[:,:,1]
        blurredimg = cv.GaussianBlur(gimg, (3,3), 0)
        circles = cv.HoughCircles(blurredimg, cv.HOUGH_GRADIENT, 1, 25, param1=50, param2=30, minRadius=5, maxRadius=30)
        for i in circles[0,:]:
            # draw the outer circle
            cv.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv.circle(img,(i[0],i[1]),2,(0,0,255),3)
        
        
        cv.imshow('detected circles',img)
        cv.waitKey(0)
        cv.destroyAllWindows()