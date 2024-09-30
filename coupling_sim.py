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
import matplotlib as mpl 
from scipy.optimize import minimize, curve_fit
from matplotlib.pyplot import gca
import pandas as pd
from matplotlib.colors import LogNorm
import seaborn as sn
from correlationcalculator import heatmap
from correlationcalculator import annotate_heatmap


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
    ax = np.zeros((size,size))
    ay = np.zeros((size,size))
    
    #iterate through all the spheres
    for i in range(size):
        for j in range(size):
            #update with trap's force
            ax[i,j] = ax[i,j] - kx[i,j] * x[i,j]
            ay[i,j] = ay[i,j] - ky[i,j] * y[i,j]

            #iterate through them all again to find force of sphere(m,n) on sphere(i,j)
            for m in range(size):
                for n in range(size):
                    #make sure not self interaction
                    if (i != m) and (j != n):
                        xdif = abs((m-i)*sep + x[m,n] - x[i,j])
                        ydif = abs((n-j)*sep + y[m,n] - y[i,j])
                        dist2 = (xdif)**2 + (ydif)**2
                        charge_amplitude = CC * charge[m,n] * charge[i,j] / dist2
                        
                        #get x, y components to find projections of the force
                        x_comp = charge_amplitude * xdif / math.sqrt(dist2)
                        y_comp = charge_amplitude * ydif / math.sqrt(dist2)
                        #determine sign (assume charge always -)
                        if i < m:
                            x_comp = -1*x_comp
                        if j < n:
                            y_comp = -1*y_comp
                        
                        ax[i,j] = ax[i,j] + x_comp
                        ay[i,j] = ay[i,j] + y_comp

    return ax, ay

@njit
def velocity_update(size, vx, vy, ax, ay, dt):
    #Updates the velocities of the spheres using given accelerations
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
    #updates the position of the spheres using given velocities
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
    #calculates the gas effects on the velocity for all the spheres
    c1 = np.exp(-gamma*dt)
    c2 = math.sqrt(1-c1*c1)*math.sqrt(kBT)
    for i in range(size):
        for j in range(size):
            R1 = np.random.normal()
            R2 = np.random.normal()

            vx[i,j] = c1*vx[i,j] + R1*c2
            vy[i,j] = c1*vy[i,j] + R2*c2
    return vx, vy


def old_random_velocity_update(v,gamma,kBT,dt):
    R = np.random.normal()
    c1 = np.exp(-gamma*dt)
    c2 = np.sqrt(1-c1*c1)*np.sqrt(kBT)
    v_new = c1*v + R*c2
    return v_new


def baoab(arraysize, timespan, dt, fs, gamma, kBT, x, y, vx, vy, kx_matrix, ky_matrix, charge_matrix, CC, sep,startrec):
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
        
        if step_number%save_frequency == 0 and t>startrec:

            for i in range(arraysize):
                for j in range(arraysize):
                    xsaves[i][j].append(x[i,j])
                    ysaves[i][j].append(y[i,j])
                    vxsaves[i][j].append(vx[i,j])
                    vysaves[i][j].append(vy[i,j])
            save_times.append(t)
        
        t = t+dt
        step_number = step_number + 1
    
    print(x)
    print(y)
    print(vx)
    print(vy)

    return save_times, xsaves, ysaves, vxsaves, vysaves

timespan = 100
dt = 0.0001
fs = 1000
#don't record the motion until t>=startrec to let the system evolve a bit
startrec = 20

arraysize = 5 #set how many rows/columns we have

#generate a list of empty lists for storing motion state
list_template = [ [ [] for i in range(arraysize)] for j in range(arraysize) ]

#numpy matrix template for storing static values
matrix_template = np.zeros((arraysize,arraysize))

pos_int_bounds = [-1E-6, 1E-6]     #starting position bounds in m
vel_ints_bounds = [0,0]            #starting velocity bounds in m/s (gonna use 0)

pos_gauss = [0,1e-6]
vel_gauss = [0,0]

pressure = 0.4      # in mbar
temp = 295          # in K
kBT = 4.073e-21     # for T = 295K (in N m)
gamma = 9.863e-10 * pressure / np.sqrt(temp)     #Epstein drag using 10um sphere (in kg/s)

#Bounds of electrons on the spheres:
charge = [500,2000]      
charge_const = 2.30708e-16      # 1 / (4 pi epsilon_0 * 1ng) in N m^2 / kg

#Resonant frequency range:
frange = [90,220]               # in Hz

x = matrix_template
y = matrix_template
vx = matrix_template
vy = matrix_template
charge_matrix = matrix_template
kx_matrix = matrix_template
ky_matrix = matrix_template

#generate random initial values for positions, velocity, spring constants and charge
rng = np.random.default_rng()

x = rng.normal(pos_gauss[0], pos_gauss[1], size = (arraysize,arraysize))
y = rng.normal(pos_gauss[0], pos_gauss[1], size = (arraysize,arraysize))
vx = rng.normal(vel_gauss[0], vel_gauss[1], size = (arraysize,arraysize))
vy = rng.normal(vel_gauss[0], vel_gauss[1], size = (arraysize,arraysize))

#Don't know the distributions of charge and k, so just doing uniform generation
charge_matrix = rng.integers(charge[0],charge[1],size = (arraysize,arraysize))   #assume all have negative charge

fx_matrix = rng.integers(frange[0],frange[1],size = (arraysize,arraysize))
fy_matrix = rng.integers(frange[0],frange[1],size = (arraysize,arraysize))
#spring constants are actually k/m
kx_matrix = freq_to_k(fx_matrix)    # in 1/s^2
ky_matrix = freq_to_k(fy_matrix)    # in 1/s^2

# for i in range(arraysize):
#     for j in range(arraysize):
        
#         #DOES NOT WORK
#         #use random triangular to get distribution weighted around 0 
#         # x[i,j] = random.triangular(pos_int_bounds[0], 0, pos_int_bounds[1])
#         # y[i,j] = random.triangular(pos_int_bounds[0], 0, pos_int_bounds[1])
#         # vx[i,j] = random.triangular(vel_ints_bounds[0], 0, vel_ints_bounds[1])
#         # vy[i,j] = random.triangular(vel_ints_bounds[0], 0, vel_ints_bounds[1])

#         #Don't know the distributions of charge and k, so just doing uniform generation
#         charge_matrix[i,j] = rng.integers(0,charge)   #assume all have negative charge
#         #spring constants are actually k/m
#         kx_matrix[i,j] = freq_to_k(rng.integers(frange[0],frange[1]))   # in 1/s^2
#         ky_matrix[i,j] = freq_to_k(rng.integers(frange[0],frange[1]))   # in 1/s^2
    

print(charge_matrix)
print(fx_matrix)
print(x)
print(vy)

sep = [55,70,85,100]          # separation in um
mpl.rcParams.update({'font.size': 18})

figa, axa = plt.subplots(1, len(sep))
cax = figa.add_axes(rect=(0.2,0.2,0.6,0.03))
#figa.suptitle('Correlation of Motion of a 5x5 Array of Spheres')
figs = {}
axs = {}
k = 0
for d in sep:
    

    save_times, xsaves, ysaves, vxsaves, vysaves  = baoab(arraysize, timespan, dt, fs, gamma, kBT,\
                                                                    x, y, vx, vy, kx_matrix, ky_matrix,\
                                                                    charge_matrix, charge_const, d*10**-6, startrec)



    # segmentsize = round(fs/2)
    # fftbinning = 1024
    # freq, PSD1 = welch(positions1[20000:], fs, 'hann', segmentsize, segmentsize/2, fftbinning, 'constant', True, 'density', 0,'mean')
    # freq, PSD2 = welch(positions2[20000:], fs, 'hann', segmentsize, segmentsize/2, fftbinning, 'constant', True, 'density', 0,'mean')
    # figs[i], axs[i] = plt.subplots()

    # axs[i].semilogy(freq,np.sqrt(PSD1))
    # axs[i].semilogy(freq,np.sqrt(PSD2))
    # axs[i].set_xlim(50,250)
    # axs[i].set_title("Simulated Coupling of Charged Spheres with Gas Interaction \n at %d um Separation and %d e Charge" % (d, charge))
    # axs[i].set_xlabel('Frequency (Hz)')
    # axs[i].set_ylabel(r'ASD ($m/\sqrt{Hz}$)')

    first = True
    for i in range(arraysize):
        for j in range(arraysize):
            if first == True:
                xarray = np.array(xsaves[i][j]).reshape(-1,1)
                yarray = np.array(ysaves[i][j]).reshape(-1,1)
                vxarray = np.array(vxsaves[i][j]).reshape(-1,1)
                vyarray = np.array(vysaves[i][j]).reshape(-1,1)
                first = False
            else:
                xarray = np.concatenate((xarray, np.array(xsaves[i][j]).reshape(-1,1)),axis=1)
                yarray = np.concatenate((yarray, np.array(ysaves[i][j]).reshape(-1,1)),axis=1)
                vxarray = np.concatenate((vxarray, np.array(vxsaves[i][j]).reshape(-1,1)),axis=1)
                vyarray = np.concatenate((vyarray, np.array(vysaves[i][j]).reshape(-1,1)),axis=1)  

    xdf = pd.DataFrame(xarray)
    ydf = pd.DataFrame(yarray)
        
    xcorrmatrix = xdf.corr()
    ycorrmatrix = ydf.corr() 

    for l in range(xcorrmatrix.shape[0]):
        for m in range(xcorrmatrix.shape[1]):

            if xcorrmatrix[l][m] == 1:
                xcorrmatrix[l][m] = 0
            if ycorrmatrix[l][m] == 1:
                ycorrmatrix[l][m] = 0

    print(np.max(xcorrmatrix))
    print(np.min(xcorrmatrix))


    spherenames = [str(x+1) for x in range(arraysize**2)]
    norm = LogNorm()
    if k == len(sep) - 1:
        plot_cbar = True
        cbar_kws = {'shrink' : 0.8,
                    'orientation': 'horizontal'}
        cbar_ax = cax
        
    else:
        plot_cbar = False
        cbar_kws = None
        cbar_ax = None
    
    symcor = xcorrmatrix
    for a in range(xcorrmatrix.shape[0]):
        for b in range(xcorrmatrix.shape[1]):
            if a < b:
                symcor[a][b] = ycorrmatrix[a][b]
    mask = np.triu(np.ones_like(xcorrmatrix, dtype=bool))
    diagmask = np.identity(xcorrmatrix.shape[0])
    sn.heatmap(symcor, mask=diagmask, square=True, cmap = 'viridis', vmin=-0.25, vmax=0.25, ax=axa[k], cbar=plot_cbar, cbar_ax = cbar_ax, cbar_kws=cbar_kws)
    axa[k].tick_params(axis='both', which='major', labelsize=18)
    #ax[i].set_xticks(np.arange(xcor.shape[1])+.5, labels=spherenames,fontsize=16)
    #ax[i].set_yticks(np.arange(xcor.shape[0])+.5, labels=spherenames,fontsize=16)
    #plt.setp(ax[i].get_xticklabels(), rotation=90)
    axa[k].set_title(str(int(d)) + r'$~\mu$m Spacing', fontsize=26, pad=15)
    axa[k].set_xlabel('Sphere Index', fontsize=22, labelpad=5)
    axa[k].set_ylabel('Sphere Index',fontsize=22,labelpad=5)

    k+=1

    
figa.tight_layout()
cax.set_title('Correlation Coefficients',fontsize=26)
plt.show()
