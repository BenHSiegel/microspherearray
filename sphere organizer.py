# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 14:26:35 2023

@author: Ben
"""
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import fft
from scipy.spatial.distance import cdist
import random


def makearray(startfreq1,startfreq2,separation,size,dim2):
    array = np.zeros((size,2))
    freq1 = startfreq1 - separation
    freq2 = startfreq2
    for i in range(size):
        
        if i%dim2 == 0:
            freq2 = startfreq2
            freq1 = freq1 + separation
            array[i,0] = freq1
            array[i,1] = freq2
        else:
            freq2 = freq2 + separation
            array[i,0] = freq1
            array[i,1] = freq2
            

    return array
        
    

# startpoints = np.arange(15,45,0.25).reshape(60,2)
# random.shuffle(startpoints[:,0])
# random.shuffle(startpoints[:,1])

startpoints = makearray(20,20,1,100,10)
random.shuffle(startpoints[:,0])
random.shuffle(startpoints[:,1])
startpoints = pd.DataFrame(startpoints)
startpoints = startpoints.drop_duplicates()
startpoints = startpoints.to_numpy()
startpoints = startpoints[:30,:]

endpoints = makearray(22, 22, 1, 30, 6)
endpoints = endpoints[:len(startpoints),:]

# endpoints = np.arange(45,15,-0.25).reshape(60,2)
# random.shuffle(endpoints[:,0]) 
# random.shuffle(endpoints[:,1])


d = cdist(startpoints, endpoints)

row_ind, col_ind = linear_sum_assignment(d)


def pathfinder(startpoints, endpoints, row_ind, col_ind):
    xtravellines = []
    ytravellines = []
    
    for i in range(0,len(startpoints)):
        
        #makes direct lines between the start and end points
        if np.abs(startpoints[row_ind[i]][0]-endpoints[col_ind[i]][0]) > np.abs(startpoints[row_ind[i]][1] - endpoints[col_ind[i]][1]):
            slope = (endpoints[col_ind[i]][1] - startpoints[row_ind[i]][1]) / (endpoints[col_ind[i]][0] - startpoints[row_ind[i]][0])
            
            if startpoints[row_ind[i]][0] > endpoints[col_ind[i]][0]:
                step = -0.05
                xsegment = np.arange(startpoints[row_ind[i]][0], endpoints[col_ind[i]][0]+step, step)
            else:
                step = 0.05
                xsegment = np.arange(startpoints[row_ind[i]][0], endpoints[col_ind[i]][0]+step, step)
            
            ysegment = slope * ( xsegment - startpoints[row_ind[i]][0] ) + startpoints[row_ind[i]][1]
            
        elif np.abs(startpoints[row_ind[i]][0]-endpoints[col_ind[i]][0]) == np.abs(startpoints[row_ind[i]][1] - endpoints[col_ind[i]][1]) == 0:
            xsegment = [ startpoints[row_ind[i]][0] ]
            ysegment = [ startpoints[row_ind[i]][1] ]
        else:
            slope = (endpoints[col_ind[i]][0] - startpoints[row_ind[i]][0]) / (endpoints[col_ind[i]][1] - startpoints[row_ind[i]][1])
            
            if startpoints[row_ind[i]][1] > endpoints[col_ind[i]][1]:
                step = -0.05
                ysegment = np.arange(startpoints[row_ind[i]][1], endpoints[col_ind[i]][1]+step, step)
            else:
                step = 0.05
                ysegment = np.arange(startpoints[row_ind[i]][1], endpoints[col_ind[i]][1]+step, step)
            
            xsegment = slope * ( ysegment - startpoints[row_ind[i]][1] ) + startpoints[row_ind[i]][0]
            
        ######################################################################
           
        # makes paths along straight lines and diagonals
        # if startpoints[row_ind[i]][0] == endpoints[col_ind[i]][0]:
        #     xsegment = [ startpoints[row_ind[i]][0] ]
        # elif startpoints[row_ind[i]][0] > endpoints[col_ind[i]][0]:
        #     step = -0.05
        #     xsegment = np.arange(startpoints[row_ind[i]][0], endpoints[col_ind[i]][0]+step, step)
        # else:
        #     step = 0.05
        #     xsegment = np.arange(startpoints[row_ind[i]][0], endpoints[col_ind[i]][0]+step, step)
        
        # if startpoints[row_ind[i]][1] == endpoints[col_ind[i]][1]:
        #     ysegment = [ startpoints[row_ind[i]][1] ]
        # elif startpoints[row_ind[i]][1] > endpoints[col_ind[i]][1]:
        #     step = -0.05
        #     ysegment = np.arange(startpoints[row_ind[i]][1], endpoints[col_ind[i]][1]+step, step)
        # else:
        #     step = 0.05
        #     ysegment = np.arange(startpoints[row_ind[i]][1], endpoints[col_ind[i]][1]+step, step)
    
        # if len(xsegment) < len(ysegment):
        #     xsegment = np.pad(xsegment,(0,(len(ysegment)-len(xsegment))),'constant',constant_values=xsegment[-1])
        # if len(ysegment) < len(xsegment):
        #     ysegment = np.pad(ysegment,(0,(len(xsegment)-len(ysegment))),'constant',constant_values=ysegment[-1])
        
        
        ######################################################################
        
        xtravellines.append(xsegment)
        ytravellines.append(ysegment)
    
        plt.plot(xsegment, ysegment,'.')
        plt.plot(xsegment[0], ysegment[0], 'og')
        plt.plot(xsegment[-1], ysegment[-1], 'ob')
    
    plt.show()
    
    return xtravellines, ytravellines

def proximitycheck(xtravellines, ytravellines, row_ind, col_ind,swapenabled):
    alarm = False
    alreadyswappedlist = []
    counter = 0;
    for i in range(max(len(a) for a in xtravellines)):
        xlocs=[]
        ylocs=[]
        for b in xtravellines:
            if i >= len(b):
                xlocs.append(b[-1])
            else:
                xlocs.append(b[i])
        for c in ytravellines:
            if i >= len(c):
                ylocs.append(c[-1])
            else:
                ylocs.append(c[i])
        
        
        for i1 in range(len(xlocs)):
            for i2 in range(len(xlocs)):
                distcheck = np.sqrt((xlocs[i2] - xlocs[i1])**2 + (ylocs[i2]-ylocs[i1])**2)
                if i1 != i2 and distcheck < 0.4 and (i1 not in alreadyswappedlist) and (i2 not in alreadyswappedlist):
                    alarm = True
                    print("Pairs "+ str(i1) + ' and '+ str(i2) + " paths overlap")
                    if swapenabled == True:
                        swap1 = col_ind[i1]
                        swap2 = col_ind[i2]
                        col_ind[i1] = swap2
                        col_ind[i2] = swap1
                    alreadyswappedlist.append(i1)
                    alreadyswappedlist.append(i2)
                    counter = counter + 1
                    
    print("There were " + str(counter) + " conflicts \n \n")                 
    return alarm, row_ind, col_ind


# def polylinepaths(xtravellines, ytravellines, row_ind, col_ind, startpoints, endpoints):
#     mindist = 0.4
#     for i in range(max(len(a) for a in xtravellines)):
#         xlocs=[]
#         ylocs=[]
#         for b in xtravellines:
#             if i >= len(b):
#                 xlocs.append(b[-1])
#             else:
#                 xlocs.append(b[i])
#         for c in ytravellines:
#             if i >= len(c):
#                 ylocs.append(c[-1])
#             else:
#                 ylocs.append(c[i])
        
        
#         for i1 in range(len(xlocs)):
#             for i2 in range(len(xlocs)):
#                 distcheck = np.sqrt((xlocs[i2] - xlocs[i1])**2 + (ylocs[i2]-ylocs[i1])**2)
#                 if i1 != i2 and distcheck < mindist and (i1 not in alreadyfixed) and (i2 not in alreadyfixed):
                                        
#                     for i3 in (i1,i2):
#                         for i4 in (i1,i2):
#                             if i4 != i3:        
#                                 if ylocs[i3] > ylocs[i4]:
#                                     disp = (mindist - (ylocs[i3] - ylocs[i4]))/2
#                                     interpoint = [xlocs[i3],ylocs[i3]+disp]
#                                 elif ylocs[i3] < ylocs[i4]:
#                                     disp = (mindist + (ylocs[i3] - ylocs[i4]))/2
#                                     interpoint = [xlocs[i3],ylocs[i3]-disp]
                        
#                         if np.abs(startpoints[row_ind[i]][0]-endpoints[col_ind[i]][0]) > np.abs(startpoints[row_ind[i]][1] - endpoints[col_ind[i]][1]):
#                             slope = (endpoints[col_ind[i]][1] - startpoints[row_ind[i]][1]) / (endpoints[col_ind[i]][0] - startpoints[row_ind[i]][0])
                            
#                             if startpoints[row_ind[i]][0] > endpoints[col_ind[i]][0]:
#                                 step = -0.05
#                                 xsegment = np.arange(startpoints[row_ind[i]][0], endpoints[col_ind[i]][0]+step, step)
#                             else:
#                                 step = 0.05
#                                 xsegment = np.arange(startpoints[row_ind[i]][0], endpoints[col_ind[i]][0]+step, step)
                            
#                             ysegment = slope * ( xsegment - startpoints[row_ind[i]][0] ) + startpoints[row_ind[i]][1]
                            
#                         elif np.abs(startpoints[row_ind[i]][0]-endpoints[col_ind[i]][0]) == np.abs(startpoints[row_ind[i]][1] - endpoints[col_ind[i]][1]) == 0:
#                             xsegment = [ startpoints[row_ind[i]][0] ]
#                             ysegment = [ startpoints[row_ind[i]][1] ]
#                         else:
#                             slope = (endpoints[col_ind[i]][0] - startpoints[row_ind[i]][0]) / (endpoints[col_ind[i]][1] - startpoints[row_ind[i]][1])
                            
#                             if startpoints[row_ind[i]][1] > endpoints[col_ind[i]][1]:
#                                 step = -0.05
#                                 ysegment = np.arange(startpoints[row_ind[i]][1], endpoints[col_ind[i]][1]+step, step)
#                             else:
#                                 step = 0.05
#                                 ysegment = np.arange(startpoints[row_ind[i]][1], endpoints[col_ind[i]][1]+step, step)
                            
#                             xsegment = slope * ( ysegment - startpoints[row_ind[i]][1] ) + startpoints[row_ind[i]][0]



def delaypath(xtravellines, ytravellines, row_ind, col_ind):
    mindist = 0.4
    alreadyfixed = []
    for i in range(max(len(a) for a in xtravellines)):
        xlocs=[]
        ylocs=[]
        for b in xtravellines:
            if i >= len(b):
                xlocs.append(b[-1])
            else:
                xlocs.append(b[i])
        for c in ytravellines:
            if i >= len(c):
                ylocs.append(c[-1])
            else:
                ylocs.append(c[i])
        
        
        for i1 in range(len(xlocs)):
            for i2 in range(len(xlocs)):
                distcheck = np.sqrt((xlocs[i2] - xlocs[i1])**2 + (ylocs[i2]-ylocs[i1])**2)
                if i1 != i2 and distcheck < mindist and (i1 not in alreadyfixed) and (i2 not in alreadyfixed):
                    if len(xtravellines[i1]) < len(xtravellines[i2]):
                        dchoice = i1
                    else:
                        dchoice = i2
                    if i > len(xtravellines[dchoice]):
                        pos = -10
                    else:
                        pos = i-2
                    xtravellines[dchoice] = np.concatenate((xtravellines[dchoice][:pos], [xtravellines[dchoice][pos]]*10, xtravellines[dchoice][pos:]))
                    ytravellines[dchoice] = np.concatenate((ytravellines[dchoice][:pos], [ytravellines[dchoice][pos]]*10, ytravellines[dchoice][pos:]))
                    alreadyfixed.append(i1)
                    alreadyfixed.append(i2)
    return xtravellines, ytravellines



xtravellines, ytravellines = pathfinder(startpoints, endpoints, row_ind, col_ind)

alarm, row_ind, col_ind = proximitycheck(xtravellines, ytravellines, row_ind, col_ind,True) 


def doublecheck(alarm,xtravellines,ytravellines,row_ind,col_ind,startpoints, endpoints):
    
    if alarm == True:
        xtravellines, ytravellines = pathfinder(startpoints, endpoints, row_ind, col_ind)
        alarm, row_ind, col_ind = proximitycheck(xtravellines, ytravellines, row_ind, col_ind,True)
   
    return alarm, xtravellines, ytravellines, row_ind, col_ind

for i in range(0,10):
    alarm, xtravellines, ytravellines, row_ind, col_ind = doublecheck(alarm, xtravellines, ytravellines, row_ind, col_ind, startpoints, endpoints)
    

if alarm == True:
    
    newxtravellines, newytravellines = delaypath(xtravellines, ytravellines, row_ind, col_ind)
    print('Delayed a path to prevent collision')
    alarm, row_ind, col_ind = proximitycheck(newxtravellines, newytravellines, row_ind, col_ind, False)
    if alarm == True:
        alarm, row_ind, col_ind = proximitycheck(xtravellines, ytravellines, row_ind, col_ind,True)
        print("Trying delay on swap")
        newxtravellines, newytravellines = delaypath(xtravellines, ytravellines, row_ind, col_ind)
        alarm, row_ind, col_ind = proximitycheck(newxtravellines, newytravellines, row_ind, col_ind,False)
    
    for i in range(len(xtravellines)):
        plt.plot(newxtravellines[i], newytravellines[i],'.')
        plt.plot(newxtravellines[i][0], newytravellines[i][0], 'og')
        plt.plot(newxtravellines[i][-1], newytravellines[i][-1], 'ob')
    plt.title('Delayed path')
    
    plt.show()
        