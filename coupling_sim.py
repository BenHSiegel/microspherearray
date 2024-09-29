'''
BAOAB coupled langevin estimator
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
import random
import numba
import math
from numba import njit

def freq_to_k(f):
    #takes in frequency of motion and gives back k/m
    k = (f*2*np.pi)**2
    return k

def motion_eq(x, x_other, k, CC):
    #Harmonic oscillator approximation ignoring the constant repulsive force
    acc = -k*x - CC*(x-x_other)
    return acc

def full_motion_eq(x, x_other, k, CC, d):
    #Non approximated force equation for two negatively charged spheres
    #direction is included in the CC term
    acc = -k*x + CC/(d+x_other-x)**2
    return acc

@njit
def force_calc(size,x,y,kx,ky,charge,CC,sep):
    #calculate the overall x and y components of the force on each sphere
    ax = np.zeros(size,size)
    ay = np.zeros(size,size)
    for i in range(size):
        for j in range(size):
            ax[i,j] = ax[i,j] - kx[i,j] * x[i,j]
            ay[i,j] = ay[i,j] - ky[i,j] * y[i,j]

            for m in range(size):
                for n in range(size):
                    if i != m and j != n:
                        xdif = abs((m-i)*sep + x[m,n] - x[i,j])
                        ydif = abs((n-j)*sep + y[m,n] - y[i,j])
                        dist2 = (xdif)**2 + (ydif)**2
                        charge_amplitude = CC * charge[m,n] * charge[i,j] / dist2
                        x_comp = charge_amplitude * xdif / math.sqrt(dist2)
                        y_comp = charge_amplitude * ydif / math.sqrt(dist2)
                        if i < m:
                            x_comp = -1*x_comp
                        if j < n:
                            y_comp = -1*y_comp
                        ax[i,j] = ax[i,j] + x_comp
                        ay[i,j] = ay[i,j] + y_comp

    return ax, ay

@njit
def velocity_update(size, vx, vy, ax, ay, dt):
    for i in range(size):
        for j in range(size):
            vx[i,j] = vx[i,j] + ax[i,j] * dt / 2.0
            vy[i,j] = vy[i,j] + ay[i,j] * dt / 2.0
    return vx, vy

def old_velocity_update(v,a,dt):
    v_new = v + a*dt/2.0
    return v_new

@njit
def position_update(size, x, y, vx, vy, dt):
    for i in range(size):
        for j in range(size):
            x[i,j] = x[i,j] + vx[i,j] * dt / 2.0
            y[i,j] = y[i,j] + vy[i,j] * dt / 2.0
    return x, y

def old_position_update(x,v,dt):
    x_new = x + v*dt/2.0
    return x_new

@njit
def random_velocity_update(size, vx, vy, gamma, kBT, dt):
    c1 = np.exp(-gamma*dt)
    c2 = math.sqrt(1-c1*c1)*math.sqrt(kBT)
    for i in range(size):
        for j in range(size):
            R = np.random.normal()
            
            #probably not great but just applying same random
            #   force to both x and y
            vx[i,j] = c1*vx[i,j] + R*c2
            vy[i,j] = c1*vy[i,j] + R*c2
    return vx, vy


def old_random_velocity_update(v,gamma,kBT,dt):
    R = np.random.normal()
    c1 = np.exp(-gamma*dt)
    c2 = np.sqrt(1-c1*c1)*np.sqrt(kBT)
    v_new = c1*v + R*c2
    return v_new


def baoab(arraysize, timespan, dt, fs, gamma, kBT, x, y, vx, vy, kx_matrix, ky_matrix, charge_matrix, CC, sep):
    save_frequency = 1/(dt*fs)
    
    t = 0
    step_number = 0
    xsaves=ysaves=vxsaves=vysaves = [ [ [] for i in range(arraysize)] for j in range(arraysize) ]
    save_times = []
    
    while(t<timespan):
        
        # B
        ax, ay = force_calc(arraysize,x,y,kx_matrix,ky_matrix,charge_matrix,CC,sep)
        vx, vy = velocity_update(arraysize, vx, vy, ax, ay, dt)
        #A
        x, y = position_update(arraysize, x, y, vx, vy, dt)
        #O
        vx, vy = random_velocity_update(arraysize, vx, vy, gamma, kBT, dt)
        #A
        x, y = position_update(arraysize, x, y, vx, vy, dt)
        # B
        ax, ay = force_calc(arraysize,x,y,kx_matrix,ky_matrix,charge_matrix,CC,sep)
        vx, vy = velocity_update(arraysize, vx, vy, ax, ay, dt)
        
        if step_number%save_frequency == 0 and step_number>0:

            for i in range(arraysize):
                for j in range(arraysize):
                    xsaves[i][j].append(x[i,j])
                    ysaves[i][j].append(y[i,j])
                    vxsaves[i][j].append(vx[i,j])
                    vysaves[i][j].append(vy[i,j])
            save_times.append(t)
        
        t = t+dt
        step_number = step_number + 1
    
    return save_times, xsaves, ysaves, vxsaves, vysaves

timespan = 50
dt = 0.0001
fs = 1000

arraysize = 5 #set how many rows/columns we have

#generate a list of empty lists for storing motion state
list_template = [ [ [] for i in range(arraysize)] for j in range(arraysize) ]

#numpy matrix template for storing static values
matrix_template = np.zeros((arraysize,arraysize))

pos_int_bounds = [-1e-6, 1e-6]     #starting position bounds in m
vel_ints_bounds = [0,0]            #starting velocity bounds in m/s (gonna use 0)

pressure = 0.4      # in mbar
temp = 295          # in K
kBT = 4.073e-21     # for T = 295K (in N m)
gamma = 9.863e-10 * pressure / np.sqrt(temp)     #Epstein drag using 10um sphere (in kg/s)

#Upper bound of electrons on the spheres:
charge = 1000                    # I doubt that we would break even 500 electron while loading with plasma       
charge_const = 2.30708e-16      # 1 / (4 pi epsilon_0 * 1ng) in N m^2 / kg

#Resonant frequency range:
frange = [90,260]               # in Hz

x = matrix_template
y = matrix_template
vx = matrix_template
vy = matrix_template
charge_matrix = matrix_template
kx_matrix = matrix_template
ky_matrix = matrix_template

random.seed(5)
for i in range(arraysize):
    for j in range(arraysize):
        
        x[i,j] = random.triangular(pos_int_bounds[0], pos_int_bounds[1], 0)
        y[i,j] = random.triangular(pos_int_bounds[0], pos_int_bounds[1], 0)
        vx[i,j] = random.triangular(vel_ints_bounds[0], vel_ints_bounds[1], 0)
        vy[i,j] = random.triangular(vel_ints_bounds[0], vel_ints_bounds[1], 0)

        charge_matrix[i,j] = random.randrange(charge)   #assume all have negative charge
        #spring constants are actually k/m
        kx_matrix[i,j] = freq_to_k(random.randrange(frange[0],frange[1]))   # in 1/s^2
        ky_matrix[i,j] = freq_to_k(random.randrange(frange[0],frange[1]))   # in 1/s^2
    


sep = [100, 85, 70, 55]          # separation in um
figs = {}
axs = {}
k = 0
for d in sep:
    

    save_times, xsaves, ysaves, vxsaves, vysaves  = baoab(arraysize, timespan, dt, fs, gamma, kBT,\
                                                                    x, y, vx, vy, kx_matrix, ky_matrix,\
                                                                    charge_matrix, CC=charge_const, sep=(d*10**-6))


    segmentsize = round(fs/2)
    fftbinning = 1024
    freq, PSD1 = welch(positions1[20000:], fs, 'hann', segmentsize, segmentsize/2, fftbinning, 'constant', True, 'density', 0,'mean')
    freq, PSD2 = welch(positions2[20000:], fs, 'hann', segmentsize, segmentsize/2, fftbinning, 'constant', True, 'density', 0,'mean')
    figs[i], axs[i] = plt.subplots()

    axs[i].semilogy(freq,np.sqrt(PSD1))
    axs[i].semilogy(freq,np.sqrt(PSD2))
    axs[i].set_xlim(50,250)
    axs[i].set_title("Simulated Coupling of Charged Spheres with Gas Interaction \n at %d um Separation and %d e Charge" % (d, charge))
    axs[i].set_xlabel('Frequency (Hz)')
    axs[i].set_ylabel(r'ASD ($m/\sqrt{Hz}$)')

    i += 1

plt.show()
