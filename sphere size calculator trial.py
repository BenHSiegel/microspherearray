import numpy as np
import cv2 as cv
import os

#Input the file path to the picture you want
path = r"D:\Lab data\20240604\0-8MHz"
filename = "pictry.png"
os.path.join(path,filename)

#Read the picture with cv2
img = cv.imread(filename)

#check there is a picture
assert img is not None, "file could not be read, check with os.path.exists()"
#convert to black and white since it doesn't like if it has blue, green, red data inputs (the camera is probably grayscale already anyways)
gimg = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#Perform a gausian blur on the image this smooths out sharp lines which usually makes feature recognition algorithms work better
#this smooths out in 3x3 square chunks
blurredimg = cv.GaussianBlur(gimg, (3,3), 0)

#Does the Hough transform to get where circles are and their size
circles = cv.HoughCircles(blurredimg, cv.HOUGH_GRADIENT, 1, 25, param1=20, param2=5, minRadius=2, maxRadius=20)
circlespx = np.uint16(np.around(circles))

#draws all the circles on the picture you gave it to check if it worked
for i in circlespx[0,:]:
    # draw the outer circle
    cv.circle(img,(i[0],i[1]),i[2],(0,255,0),3)
    # draw the center of the circle
    cv.circle(img,(i[0],i[1]),2,(0,0,255),1)
    
#show the picture and waits till you hit a key to close the picture window
cv.imshow('detected circles',img)
cv.waitKey(0)
cv.destroyAllWindows()
