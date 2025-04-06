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
from scipy.spatial.distance import cdist
import random
import os
import sys



def makearray(startfreq1,startfreq2,separation,size,dim2):
    '''
    makearray will generate a list of points to use as the final array locations
    The lower frequencies in the array are defined by startfreq1 and startfreq2
    Separation : float
        defines how far apart you want the array spots to be in MHz (note
     1 MHz ~ 70um)
    size : float 
        defines the number of points in the array
    dim2 : float
        defines how many rows will be in the array
    It will make a rectangular array with set number of rows and as many columns 
    as needed. It will half fill columns starting at the top for the final spots
    '''
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
        

# endpoints = makearray(22, 22, 1, 100, 6)
# endpoints = endpoints[:len(startpoints),:]

def optimalassignment(startpoints, endpoints):
    '''
    Matches the starting points to the ending points in a way that minimizes the 
    total distance of the differences between them
    
    Parameters
    ----------
    startpoints : 2d array of floats
        Array of all the starting points. In the format of number of points x 2
        to be a list of all the points (ch0 freq, ch1 freq)
    endpoints : 2d array of floats
        Array of all the ending points. In the format of number of points x 2
        to be a list of all the points (ch0 freq, ch1 freq)

    Returns
    -------
    xtravellines : 1d array of array of floats
        DESCRIPTION.
    ytravellines : TYPE
        DESCRIPTION.
    row_ind : TYPE
        DESCRIPTION.
    col_ind : TYPE
        DESCRIPTION.

    '''
    d = cdist(startpoints, endpoints)

    row_ind, col_ind = linear_sum_assignment(d)
    
    xtravellines = [ [] for i in range(len(startpoints)) ]
    ytravellines = [ [] for i in range(len(startpoints)) ]
    
    return xtravellines, ytravellines, row_ind, col_ind


def pathfinder(xtravellines, ytravellines, startpoints, endpoints, row_ind, col_ind, todrawlist):
    '''
    pathfinder takes in the starting points of the spheres and desired endpoints
    for the specified pairs
    It then will make direct lines between them

    Parameters
    ----------
    xtravellines : TYPE
        DESCRIPTION.
    ytravellines : TYPE
        DESCRIPTION.
    startpoints : TYPE
        DESCRIPTION.
    endpoints : TYPE
        DESCRIPTION.
    row_ind : TYPE
        DESCRIPTION.
    col_ind : TYPE
        DESCRIPTION.
    todrawlist : TYPE
        DESCRIPTION.

    Returns
    -------
    xtravellines : TYPE
        DESCRIPTION.
    ytravellines : TYPE
        DESCRIPTION.

    '''
    for i in todrawlist:
        
        #makes direct lines between the start and end points
        if np.abs(startpoints[row_ind[i]][0]-endpoints[col_ind[i]][0]) > np.abs(startpoints[row_ind[i]][1] - endpoints[col_ind[i]][1]):
            slope = (endpoints[col_ind[i]][1] - startpoints[row_ind[i]][1]) / (endpoints[col_ind[i]][0] - startpoints[row_ind[i]][0])
            
            if startpoints[row_ind[i]][0] > endpoints[col_ind[i]][0]:
                step = -0.05
                xsegment = np.arange(startpoints[row_ind[i]][0], endpoints[col_ind[i]][0]-step, step)
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
                ysegment = np.arange(startpoints[row_ind[i]][1], endpoints[col_ind[i]][1]-step, step)
            else:
                step = 0.05
                ysegment = np.arange(startpoints[row_ind[i]][1], endpoints[col_ind[i]][1]+step, step)
            
            xsegment = slope * ( ysegment - startpoints[row_ind[i]][1] ) + startpoints[row_ind[i]][0]
            
        ######################################################################
           
        # makes paths along straight lines and perfect diagonals (slope -1 or 1)
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
        #adds the endpoints to the end of the paths to ensure that it ends at the desired spots
        xsegment = np.append(xsegment,endpoints[col_ind[i]][0])
        ysegment = np.append(ysegment,endpoints[col_ind[i]][1])
        xsegment = np.round(xsegment, 4)
        ysegment = np.round(ysegment, 4)
        xtravellines[i] = xsegment
        ytravellines[i] = ysegment
    
    
    return xtravellines, ytravellines


def proximitycheck(xtravellines, ytravellines, row_ind, col_ind, swapenabled):
    '''
    
    '''
    
    alarm = False
    alreadyswappedlist = []
    counter = 0
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
                    #print("Pairs "+ str(i1) + ' and '+ str(i2) + " paths overlap")
                    if swapenabled == True:
                        swap1 = col_ind[i1]
                        swap2 = col_ind[i2]
                        col_ind[i1] = swap2
                        col_ind[i2] = swap1
                    alreadyswappedlist.append(i1)
                    alreadyswappedlist.append(i2)
                    counter = counter + 1
                    
    #print("There were " + str(counter) + " conflicts \n \n")                 
    return alarm, row_ind, col_ind, counter, alreadyswappedlist


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



def delaypath(xtravellines, ytravellines, row_ind, col_ind, delaylist):
    '''
    '''
    
    mindist = 0.4
    alreadyfixed = []
    for i in range(max(len(a) for a in xtravellines)):
        
        xlocs=[]
        ylocs=[]
        
        for b in delaylist:
            if i >= len(xtravellines[b]):
                xlocs.append(xtravellines[b][-1])
            else:
                xlocs.append(xtravellines[b][i])
                
        for c in delaylist:
            if i >= len(ytravellines[c]):
                ylocs.append(ytravellines[c][-1])
            else:
                ylocs.append(ytravellines[c][i])
        
        
        for i1 in range(len(xlocs)):
            for i2 in range(len(xlocs)):
                distcheck = np.sqrt((xlocs[i2] - xlocs[i1])**2 + (ylocs[i2]-ylocs[i1])**2)
                if i1 != i2 and distcheck < mindist and (i1 not in alreadyfixed) and (i2 not in alreadyfixed):
                    if len(xtravellines[delaylist[i1]]) < len(xtravellines[delaylist[i2]]):
                        dchoice = delaylist[i1]
                        longer = delaylist[i2]
                    else:
                        dchoice = delaylist[i2]
                        longer = delaylist[i1]
                        
                    if i > len(xtravellines[dchoice]):
                        if 9 > len(xtravellines[dchoice]):
                            pos = -2
                        else:
                            pos = -10
                        #if performance is bad, change delaylength to just 10 for all cases
                        delaylength = (i - len(xtravellines[dchoice])) + 10
                    else:
                        pos = i-2
                        delaylength = 10
                        
                    xtest = np.concatenate((xtravellines[dchoice][:pos], [xtravellines[dchoice][pos]]*delaylength, xtravellines[dchoice][pos:]))
                    ytest = np.concatenate((ytravellines[dchoice][:pos], [ytravellines[dchoice][pos]]*delaylength, ytravellines[dchoice][pos:]))
                    #xtest = np.concatenate(([xtravellines[dchoice][0]]*30, xtravellines[dchoice]))
                    #ytest = np.concatenate(([ytravellines[dchoice][0]]*30, ytravellines[dchoice]))
                    delaysuccess = True
                    for k in range(len(xtravellines[longer])):
                        if k >= len(xtest):
                            distcheck2 = np.sqrt((xtest[-1] - xtravellines[longer][k])**2 + (ytest[-1]-ytravellines[longer][k])**2)
                        else:
                            distcheck2 = np.sqrt((xtest[k] - xtravellines[longer][k])**2 + (ytest[k]-ytravellines[longer][k])**2)
                        if distcheck2 < mindist:
                            delaysuccess = False
                    
                    if delaysuccess == True:
                        xtravellines[dchoice] = xtest
                        ytravellines[dchoice] = ytest
                    alreadyfixed.append(i1)
                    alreadyfixed.append(i2)
    return xtravellines, ytravellines


def doublecheck(alarm,xtravellines,ytravellines,row_ind,col_ind,startpoints, endpoints, redrawlist):
    '''
    
    '''
    
    for i in range(0,10):
    
        if alarm == True:
            xtravellines, ytravellines = pathfinder(xtravellines, ytravellines, startpoints, endpoints, row_ind, col_ind, redrawlist)
            alarm, row_ind, col_ind, counter, redrawlist = proximitycheck(xtravellines, ytravellines, row_ind, col_ind,True)
    trieddelay = False
    delaycounter = counter
    
    for i in range(0,3):
        if alarm == True:
            lastcounter = delaycounter
            trieddelay = True
            xtravellines, ytravellines = pathfinder(xtravellines, ytravellines, startpoints, endpoints, row_ind, col_ind, redrawlist)
            newxtravellines, newytravellines = delaypath(xtravellines, ytravellines, row_ind, col_ind, redrawlist)
            alarm, row_ind, col_ind, delaycounter, redrawlist = proximitycheck(newxtravellines, newytravellines, row_ind, col_ind, True)
            if delaycounter < lastcounter:
                xtravellines = newxtravellines
                ytravellines = newytravellines
            else:
                delaycounter = lastcounter
    return xtravellines, ytravellines, row_ind, col_ind, counter, trieddelay, delaycounter, redrawlist

###############################################################################
'''
Run this section to generate a file to load into labview with the sorting paths
'''



path = r"C:\Users\yalem\OneDrive\Documents\Optlev\LabView Code\ARRAY\Organizer startpoints and path csv"
os.chdir(path)

filename = r"\startpoints.csv"
filename = path + filename

startpoints = np.genfromtxt(filename,delimiter=',')

startfreq0 = float(sys.argv[1])
startfreq1 = float(sys.argv[2])
col_length = int(sys.argv[3])

endpoints = makearray(startfreq0, startfreq1, 1, len(startpoints), col_length)

counter = 0
trieddelay = False
delaycounter = 0

xtravellines, ytravellines, row_ind, col_ind = optimalassignment(startpoints, endpoints)

xtravellines, ytravellines = pathfinder(xtravellines, ytravellines, startpoints, endpoints, row_ind, col_ind, np.arange(0,(len(startpoints))))

alarm, row_ind, col_ind, counter, redrawlist = proximitycheck(xtravellines, ytravellines, row_ind, col_ind,True)

if alarm == True:
    xtravellines, ytravellines, row_ind, col_ind, counter, trieddelay, delaycounter, badpaths = doublecheck(alarm, xtravellines, ytravellines, row_ind, col_ind, startpoints, endpoints, redrawlist)

print(delaycounter)

for i in range(len(xtravellines)):
    plt.plot(xtravellines[i], ytravellines[i],'.')
    plt.plot(xtravellines[i][0], ytravellines[i][0], 'og')
    plt.plot(xtravellines[i][-1], ytravellines[i][-1], 'ob')

for i in badpaths:
    lines = plt.plot(xtravellines[i], ytravellines[i],'--')
    plt.setp(lines, color='black')
    plt.plot(xtravellines[i][0], ytravellines[i][0], 'rs')
    plt.plot(xtravellines[i][-1], ytravellines[i][-1], 'r^')
    
plt.grid()
    
plt.title('Sorting paths')

plt.show()

fixedrowsize = max(len(a) for a in xtravellines)
for i in range(len(xtravellines)):
    comparelength = len(xtravellines[i])
    if comparelength < fixedrowsize:
        padding = fixedrowsize - comparelength
        xtravellines[i] = np.concatenate((xtravellines[i], [xtravellines[i][-1]]*padding))
        

fixedrowsize = max(len(a) for a in ytravellines)        
for i in range(len(ytravellines)):
    comparelength = len(ytravellines[i])
    if comparelength < fixedrowsize:
        padding = fixedrowsize - comparelength
        ytravellines[i] = np.concatenate((ytravellines[i], [ytravellines[i][-1]]*padding))

#sanitize folder to make sure there isn't junk from previous sortings
try:
    os.remove("ch0paths.csv")
except OSError:
    pass

try:
    os.remove("ch1paths.csv")
except OSError:
    pass
        
np.savetxt("ch0paths.csv", xtravellines, delimiter=",", fmt='%1.4f')
np.savetxt("ch1paths.csv", ytravellines, delimiter=",", fmt='%1.4f')


###############################################################################
#Testing code for efficiency of sorting and brute force finding errors

# numspheres = [20,30,49,75,81,100]

# freqset = [22, 22, 21, 20, 20, 20]
# arraysize = [5,5,7,8,9,10]

# countertrials = [ [] for i in range(len(numspheres)) ]
# delaytrials = [ [] for i in range(len(numspheres)) ]
# delaycountertrials = [ [] for i in range(len(numspheres)) ]
# countertrialsavg = []
# delaycountertrialsavg = []

# for select in range(len(numspheres)):
    
#     print('Doing ' + str(numspheres[select]) + ' spheres')
    
#     for t in range(1000):
        
#         counter = 0
#         trieddelay = False
#         delaycounter = 0
        
#         startpoints = makearray(17,17,1,256,16)
#         random.shuffle(startpoints[:,0])
#         random.shuffle(startpoints[:,1])
#         startpoints = pd.DataFrame(startpoints)
#         startpoints = startpoints.drop_duplicates()
#         startpoints = startpoints.to_numpy()
#         startpoints = startpoints[:numspheres[select],:]
        
#         endpoints = makearray(freqset[select], freqset[select], 1, numspheres[select], arraysize[select])
#         endpoints = endpoints[:len(startpoints),:]
        
#         xtravellines, ytravellines, row_ind, col_ind = optimalassignment(startpoints, endpoints)
        
#         xtravellines, ytravellines = pathfinder(xtravellines, ytravellines, startpoints, endpoints, row_ind, col_ind, np.arange(0,(len(startpoints))))
        
#         alarm, row_ind, col_ind, counter, redrawlist = proximitycheck(xtravellines, ytravellines, row_ind, col_ind,True)
        
#         if alarm == True:
#             xtravellines, ytravellines, row_ind, col_ind, counter, trieddelay, delaycounter, redrawlist = doublecheck(alarm, xtravellines, ytravellines, row_ind, col_ind, startpoints, endpoints, redrawlist)
    
#         countertrials[select].append(counter)
#         delaytrials[select].append(trieddelay)
#         delaycountertrials[select].append(delaycounter)
        
    
#     countertrialsavg.append(np.mean(countertrials[select]))
#     delaycountertrialsavg.append(np.mean(delaycountertrials[select]))
    
# delaysums = []
# for i in delaytrials:
#     delaysums.append(sum(i))


# fig1, ax1 = plt.subplots()
# ax1.plot([str(j) for j in numspheres], countertrialsavg, '.')
# ax1.plot([str(j) for j in numspheres], delaycountertrialsavg, '.')
# ax1.grid(axis='y')
# ax1.set_xlabel('Spheres to Rearrange')
# ax1.set_ylabel('Average Number of Collisions')
# ax1.legend(['Swapping', 'Swapping + Delay'])
# ax1.set_title('Collision per number of spheres for 1000 simulations')


# fig3, ax3 = plt.subplots()
# ax3.plot([str(j) for j in numspheres], delaysums, '.')
# ax3.set_xlabel('Spheres to Rearrange')
# ax3.set_ylabel('Number of times delayed')
# ax3.set_title('Attempts to delay path per number of spheres for 1000 simulations')

# figs={}
# axs={}
# for i in range(len(countertrials)):
    
#     figs[i], axs[i] = plt.subplots(1, 2, sharey=True, tight_layout=True) 

#     binning = max(countertrials[i])
#     axs[i][0].hist(countertrials[i], binning)
#     figs[i].suptitle("Histogram of number of expected collisions for rearranging " + str(numspheres[i]) + ' spheres')
#     axs[i][0].set_xlabel('Number of collisions')
#     axs[i][0].set_ylabel('Frequency in 1000 simulation trials')
#     axs[i][0].set_title('Swapping method')
    
#     binning = max(delaycountertrials[i])
#     axs[i][1].hist(delaycountertrials[i], binning)
#     axs[i][1].set_title('Swap and delay method')
#     axs[i][1].set_xlabel('Number of collisions')
#     plt.show()

