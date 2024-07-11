'''
BAOAB coupled langevin estimator
'''

import numpy as np
import matplotlib.pyplot as plt
import math
import random
from tkinter import *
from array import *



def motion_eq(x, x_other, k, CC):
    acc = -k*x - CC*(x-x_other)
    return acc


def position_update(x,v,dt):
    x_new = x + v*dt/2.0
    return x_new

def velocity_update(v,a,dt):
    v_new = v + a*dt/2.0
    return v_new

def random_velocity_update(v,gamma,kBT,dt):
    R = np.random.normal()
    c1 = np.exp(-gamma*dt)
    c2 = np.sqrt(1-c1*c1)*np.sqrt(kBT)
    v_new = c1*v + R*c2
    return v_new

def baoab(motion_eq, timespan, dt, fs, gamma, kBT, pos_init1, pos_init2, vel_init1, vel_init2, k1, k2, CC):
    
    save_frequency = 1/(dt*fs)
    
    x1 = pos_init1
    v1 = vel_init1
    x2 = pos_init2
    v2 = vel_init2
    t = 0
    step_number = 0
    positions1 = []
    velocities1 = []
    positions2 = []
    velocities2 = []
    save_times = []
    
    while(t<timespan):
        
        # B
        force1 = motion_eq(x1,x2,k1,CC)
        force2 = motion_eq(x2,x1,k2,CC)
        v1 = velocity_update(v1,force1,dt)
        v2 = velocity_update(v2,force2,dt)
        #A
        x1 = position_update(x1,v1,dt)
        x2 = position_update(x2,v2,dt)
        #O
        v1 = random_velocity_update(v1,gamma,kBT,dt)
        v2 = random_velocity_update(v2,gamma,kBT,dt)
        #A
        x1 = position_update(x1,v1,dt)
        x2 = position_update(x2,v2,dt)
        # B
        force1 = motion_eq(x1,x2,k1,CC)
        force2 = motion_eq(x2,x1,k2,CC)
        v1 = velocity_update(v1,force1,dt)
        v2 = velocity_update(v2,force2,dt)
        
        if step_number%save_frequency == 0 and step_number>0:

            positions1.append(x1)
            velocities1.append(v1)
            positions2.append(x2)
            velocities2.append(v2)
            save_times.append(t)
        
        t = t+dt
        step_number = step_number + 1
    
    return save_times, positions1, velocities1, positions2, velocities2 

timespan = 30
dt = 0.001
fs = 600

pos_ints = [1e-6,-5e-7]
vel_ints = [0,0]

gamma = 
kBT = 
k1 = 130
k2 = 150
charge_coupling = 

times, positions1, velocities1, positions2, velocities2  = baoab(motion_eq, timespan, dt, fs, gamma, kBT,\
                                                                    pos_init1=pos_ints[0], pos_init2=pos_ints[1],\
                                                                    vel_init1=vel_ints[0], vel_init2=vel_ints[1],\
                                                                    k1=k1, k2=k2, CC=charge_coupling)