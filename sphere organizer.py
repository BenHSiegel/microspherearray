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

# endpoints = np.arange(45,15,-0.25).reshape(60,2)
# random.shuffle(endpoints[:,0]) 
# random.shuffle(endpoints[:,1])

# startpoints = makearray(20,20,1,100,10)
# random.shuffle(startpoints[:,0])
# random.shuffle(startpoints[:,1])
# startpoints = pd.DataFrame(startpoints)
# startpoints = startpoints.drop_duplicates()
# startpoints = startpoints.to_numpy()
# startpoints = startpoints[:30,:]

# endpoints = makearray(22, 22, 1, 100, 6)
# endpoints = endpoints[:len(startpoints),:]

def optimalassignment(startpoints, endpoints):

    d = cdist(startpoints, endpoints)

    row_ind, col_ind = linear_sum_assignment(d)
    
    return row_ind, col_ind


def pathfinder(startpoints, endpoints, row_ind, col_ind):
    xtravellines = []
    ytravellines = []
    
    for i in range(0,len(startpoints)):
        
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
        
        xsegment = np.append(xsegment,endpoints[col_ind[i]][0])
        ysegment = np.append(ysegment,endpoints[col_ind[i]][1])
        xtravellines.append(xsegment)
        ytravellines.append(ysegment)
    
    
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
    return alarm, row_ind, col_ind, counter


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
                        if 10 > len(xtravellines[dchoice]):
                            pos = -2
                        else:
                            pos = -10
                    else:
                        pos = i-2
                    xtravellines[dchoice] = np.concatenate((xtravellines[dchoice][:pos], [xtravellines[dchoice][pos]]*10, xtravellines[dchoice][pos:]))
                    ytravellines[dchoice] = np.concatenate((ytravellines[dchoice][:pos], [ytravellines[dchoice][pos]]*10, ytravellines[dchoice][pos:]))
                    alreadyfixed.append(i1)
                    alreadyfixed.append(i2)
    return xtravellines, ytravellines
 


def doublecheck(alarm,xtravellines,ytravellines,row_ind,col_ind,startpoints, endpoints):
    
    for i in range(0,10):
    
        if alarm == True:
            xtravellines, ytravellines = pathfinder(startpoints, endpoints, row_ind, col_ind)
            alarm, row_ind, col_ind, counter = proximitycheck(xtravellines, ytravellines, row_ind, col_ind,True)
    
    trieddelay = False
    flippeddelay = False
    delaycounter = 0
    if alarm == True:
       xtravellines, ytravellines = delaypath(xtravellines, ytravellines, row_ind, col_ind)
       trieddelay = True
       alarm, row_ind, col_ind, delaycounter = proximitycheck(xtravellines, ytravellines, row_ind, col_ind, True)
       if alarm == True:
           flippeddelay = True
           xtravellines, ytravellines = pathfinder(startpoints, endpoints, row_ind, col_ind)
           xtravellines, ytravellines = delaypath(xtravellines, ytravellines, row_ind, col_ind)
           alarm, row_ind, col_ind, delaycounter = proximitycheck(xtravellines, ytravellines, row_ind, col_ind,False)
    return xtravellines, ytravellines, row_ind, col_ind, counter, trieddelay, flippeddelay, delaycounter



numspheres = [20,30,49,75,81,100,125]

freqset = [22, 22, 21, 20, 20, 20, 19]
arraysize = [5,5,7,8,9,10,12]

countertrials = [ [] for i in range(len(numspheres)) ]
delaytrials = [ [] for i in range(len(numspheres)) ]
flippedtrials = [ [] for i in range(len(numspheres)) ]
delaycountertrials = [ [] for i in range(len(numspheres)) ]
countertrialsavg = []
delaycountertrialsavg = []

for select in range(len(numspheres)):
    for t in range(1000):
        
    
        counter = 0
        trieddelay = False
        flippeddelay = False
        delaycounter = 0
        
        startpoints = makearray(17,17,1,256,16)
        random.shuffle(startpoints[:,0])
        random.shuffle(startpoints[:,1])
        startpoints = pd.DataFrame(startpoints)
        startpoints = startpoints.drop_duplicates()
        startpoints = startpoints.to_numpy()
        startpoints = startpoints[:numspheres[select],:]
        
        endpoints = makearray(freqset[select], freqset[select], 1, numspheres[select], arraysize[select])
        endpoints = endpoints[:len(startpoints),:]
        
        row_ind, col_ind = optimalassignment(startpoints, endpoints)
        
        xtravellines, ytravellines = pathfinder(startpoints, endpoints, row_ind, col_ind)
        
        alarm, row_ind, col_ind, counter = proximitycheck(xtravellines, ytravellines, row_ind, col_ind,True)
        
        if alarm == True:
            xtravellines, ytravellines, row_ind, col_ind, counter, trieddelay, flippeddelay, delaycounter = doublecheck(alarm, xtravellines, ytravellines, row_ind, col_ind, startpoints, endpoints)
    
        countertrials[select].append(counter)
        delaytrials[select].append(trieddelay)
        flippedtrials[select].append(flippeddelay)
        delaycountertrials[select].append(delaycounter)
        
    
    countertrialsavg.append(np.mean(countertrials[select]))
    delaycountertrialsavg.append(np.mean(delaycountertrials[select]))
    
delaysums = []
flippedsums = []
for i in delaytrials:
    delaysums.append(i.sum())
for i in flippedtrials:
    flippedsums.append(i.sum())

fig1, ax1 = plt.subplot()
ax1.plot([str(j) for j in numspheres], countertrialsavg)
ax1.set_xlabel('Spheres to Rearrange')
ax1.set_ylabel('Average Number of Collisions')
ax1.set_title('Collision per number of spheres for 1000 simulations')

fig2, ax2 = plt.subplot()
ax2.plot([str(j) for j in numspheres], delaycountertrialsavg)
ax2.set_xlabel('Spheres to Rearrange')
ax2.set_ylabel('Average Number of Collisions')
ax2.set_title('Collision per number of spheres with delays, for 1000 simulations')

fig3, ax3 = plt.subplot()
ax3.plot([str(j) for j in numspheres], delaysums)
ax3.plot([str(j) for j in numspheres], flippedsums)
ax3.legend(['Delay tried', 'Flipped delay tried'])
ax3.set_xlabel('Spheres to Rearrange')
ax3.set_ylabel('Number of times delayed')
ax3.set_title('Attempts to delay path per number of spheres for 1000 simulations')

figs={}
axs={}
for i in range(len(countertrials)):
    
    figs[i], axs[i] = plt.subplots(1, 2, sharey=True, tight_layout=True) 

    binning = max(countertrials[i])
    axs[i][0].hist(countertrials[i], binning)
    figs[i].suptitle("Histogram of number of expected collisions for rearranging " + str(numspheres[i]) + ' spheres')
    axs[i][0].set_xlabel('Number of collisions')
    axs[i][0].set_ylabel('Frequency in 1000 simulation trials')
    axs[i][0].set_title('Swapping method')
    
    binning = max(delaycountertrials[i])
    axs[i][1].hist(delaycountertrials[i], binning)
    axs[i][1].set_title('Swap and delay method')
    axs[i][1].set_xlabel('Number of collisions')
    plt.show()
