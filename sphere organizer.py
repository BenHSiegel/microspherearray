# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 14:26:35 2023

@author: Ben
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import fft
from scipy.spatial.distance import cdist
import random

startpoints = np.arange(15,45,0.5).reshape(30,2)
random.shuffle(startpoints[:,0])
random.shuffle(startpoints[:,1])

endpoints = np.arange(45,15,-0.5).reshape(30,2)
random.shuffle(endpoints[:,0]) 
random.shuffle(endpoints[:,1])


d = cdist(startpoints, endpoints)

row_ind, col_ind = linear_sum_assignment(d)

d[row_ind,col_ind].sum()


def pathfinder(startpoints, endpoints, row_ind, col_ind):
    xtravellines = []
    ytravellines = []
    
    for i in range(0,len(startpoints)):
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
            
            xsegment = slope * ( ysegment - startpoints[row_ind[i]][1] ) + startpoints[row_ind[i]][1]
            
            
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
        
        xtravellines.append(xsegment)
        ytravellines.append(ysegment)
    
        plt.plot(xsegment, ysegment,'.')
        plt.plot(xsegment[0], ysegment[0], 'og')
        plt.plot(xsegment[-1], ysegment[-1], 'ob')
    
    plt.show()
    
    return [xtravellines, ytravellines]

def proximitycheck(xtravellines, ytravellines, row_ind, col_ind)
    alarm = False
    for i in range(max(len(a) for a in xtravellines)):
        xlocs = [xtravellines[b][i] for b in len(xtravellines)]
        ylocs = [ytravellines[b][i] for b in len(ytravellines)]
        for i1 in len(xlocs):
            for i2 in len(xlocs):
                if i1 != i2 and cdist([xlocs[i1],ylocs[i1]],[xlocs[i2],ylocs[i2]]) < 0.35:
                    alarm = True
                    print("Pairs "+ str(i1) + 'and '+ str(i2) + " paths overlap")
                    
                    
    return alarm, row_ind, col_ind