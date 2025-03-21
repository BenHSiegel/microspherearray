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
from scipy.signal import find_peaks, butter, lfilter, csd, coherence
import seaborn as sn
from correlationcalculator import heatmap
from correlationcalculator import annotate_heatmap

def lorentzian(f, f_0, T, gamma):
    kb = 1.38e-23 # Boltzmann's constant, SI units
    m = 1e-12 # mass in kg
    omega = 2*np.pi*f
    omega0 = 2*np.pi*f_0
    return kb*T/(np.pi * m) * gamma/((omega0**2 - omega**2)**2 + omega**2 * gamma**2)

def freq_to_k(f):
    #takes in frequency of motion and gives back k/m
    k = (f*2*np.pi)**2
    return k

# def motion_eq(x, x_other, k, CC):
#     #Harmonic oscillator approximation ignoring the constant repulsive force
#     acc = -k*x - CC*(x-x_other)
#     return acc

# def full_motion_eq(x, x_other, k, CC, d):
#     #Non approximated force equation for two negatively charged spheres
#     #direction is included in the CC term
#     acc = -k*x + CC/(d+x_other-x)**2
#     return acc

@njit
def force_calc(size,x,y,kx,ky,charge,CC,sep):
    #calculate the overall x and y components of the force on each sphere
    ax = np.zeros((size[0],size[1]))
    ay = np.zeros((size[0],size[1]))
    
    #iterate through all the spheres
    for i in range(size[0]):
        for j in range(size[1]):
            #update with trap's force
            ax[i,j] = ax[i,j] - kx[i,j] * x[i,j]
            ay[i,j] = ay[i,j] - ky[i,j] * y[i,j]

            #iterate through them all again to find force of sphere(m,n) on sphere(i,j)
            for m in range(size[0]):
                for n in range(size[1]):
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
    for i in range(size[0]):
        for j in range(size[1]):
            vx[i,j] = vx[i,j] + ax[i,j] * dt / 2.0
            vy[i,j] = vy[i,j] + ay[i,j] * dt / 2.0
    return vx, vy

# def old_velocity_update(v,a,dt):
#     v_new = v + a*dt/2.0
#     return v_new

@njit
def position_update(size, x, y, vx, vy, dt):
    #updates the position of the spheres using given velocities
    for i in range(size[0]):
        for j in range(size[1]):
            x[i,j] = x[i,j] + vx[i,j] * dt / 2.0
            y[i,j] = y[i,j] + vy[i,j] * dt / 2.0
    return x, y

# def old_position_update(x,v,dt):
#     x_new = x + v*dt/2.0
#     return x_new

@njit
def random_velocity_update(size, vx, vy, gamma, kBT, dt):
    #calculates the gas effects on the velocity for all the spheres
    c1 = np.exp(-gamma*dt)
    c2 = math.sqrt(1-c1*c1)*math.sqrt(kBT)
    for i in range(size[0]):
        for j in range(size[1]):
            R1 = np.random.normal()
            R2 = np.random.normal()

            vx[i,j] = c1*vx[i,j] + R1*c2
            vy[i,j] = c1*vy[i,j] + R2*c2
    return vx, vy


# def old_random_velocity_update(v,gamma,kBT,dt):
#     R = np.random.normal()
#     c1 = np.exp(-gamma*dt)
#     c2 = np.sqrt(1-c1*c1)*np.sqrt(kBT)
#     v_new = c1*v + R*c2
#     return v_new

def baoab(arraysize, timespan, dt, fs, gamma, kBT, x, y, vx, vy, kx_matrix, ky_matrix, charge_matrix, CC, sep,startrec):
    save_frequency = 1/(dt*fs)
    
    t = 0
    step_number = 0
    xsaves = [ [ [] for i in range(arraysize[0])] for j in range(arraysize[1]) ]
    ysaves = [ [ [] for i in range(arraysize[0])] for j in range(arraysize[1]) ]
    vxsaves = [ [ [] for i in range(arraysize[0])] for j in range(arraysize[1]) ]
    vysaves = [ [ [] for i in range(arraysize[0])] for j in range(arraysize[1]) ]
    save_times = []
    
    while(t<=timespan):
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
        
        if step_number%save_frequency == 0 and t>=startrec:

            for i in range(arraysize[0]):
                for j in range(arraysize[1]):
                    xsaves[j][i].append(x[i,j])
                    ysaves[j][i].append(y[i,j])
                    vxsaves[j][i].append(vx[i,j])
                    vysaves[j][i].append(vy[i,j])
            save_times.append(t)
            if len(xsaves[0][1]) != len(save_times):
                print('Mismatch')
                break
                  
        
        t = t+dt
        step_number = step_number + 1
    

    return save_times, xsaves, ysaves, vxsaves, vysaves

timespan = 50.0
dt = 0.0001
fs = 1000
#don't record the motion until t>=startrec to let the system evolve a bit
startrec = 20.0

arraysize = [2,1] #set how many rows/columns we have

#generate a list of empty lists for storing motion state
list_template = [ [ [] for i in range(arraysize[0])] for j in range(arraysize[1]) ]

#numpy matrix template for storing static values
matrix_template = np.zeros((arraysize[0],arraysize[1]))

pos_int_bounds = [-3E-6, 3E-6]     #starting position bounds in m
vel_ints_bounds = [0,0]            #starting velocity bounds in m/s (gonna use 0)

pos_gauss = [0,5e-6]
vel_gauss = [0,0]

pressure = 0.4      # in mbar
temp = 295          # in K
kBT = 4.073e-21    # 1.381e-19 for T=10000K ||4.073e-21 for T = 295K (in N m)
gamma = 9.863e-10 * pressure / np.sqrt(temp) / 1e-12             #9.863e-10 * pressure / np.sqrt(temp)     #Epstein drag using 10um sphere (in 1/s)

#Bounds of electrons on the spheres:
charge = [1000,5000]      
charge_const = 2.30708e-16      # 1 / (4 pi epsilon_0 * 1ng) in N m^2 / kg

#Resonant frequency range:
frange = [100,200]               # in Hz

x = matrix_template
y = matrix_template
vx = matrix_template
vy = matrix_template
charge_matrix = matrix_template
kx_matrix = matrix_template
ky_matrix = matrix_template

#generate random initial values for positions, velocity, spring constants and charge
rng = np.random.default_rng()

x = rng.normal(pos_gauss[0], pos_gauss[1], size = (arraysize[0],arraysize[1]))
y = rng.normal(pos_gauss[0], pos_gauss[1], size = (arraysize[0],arraysize[1]))
vx = rng.normal(vel_gauss[0], vel_gauss[1], size = (arraysize[0],arraysize[1]))
vy = rng.normal(vel_gauss[0], vel_gauss[1], size = (arraysize[0],arraysize[1]))

#Don't know the distributions of charge and k, so just doing uniform generation
fx_matrix = rng.integers(frange[0],frange[1],size = (arraysize[0],arraysize[1]))
fy_matrix = rng.integers(frange[0],frange[1],size = (arraysize[0],arraysize[1]))
#spring constants are actually k/m
kx_matrix = freq_to_k(fx_matrix)    # in 1/s^2
ky_matrix = freq_to_k(fy_matrix)    # in 1/s^2

charge_matrix = rng.integers(charge[0],charge[1],size = (arraysize[0],arraysize[1]))   #assume all have negative charge
#ALTERNATIVE: Set the charges by hand
#list of charges so you don't have to make sure the matrix is the right shape
#set_charge_list = [30, 50]
#k = 0
#for i in range(arraysize[0]):
#   for j in range(arraysize[1]):        
#       charge_matrix[i,j] = set_charge_list[k]
#       k += 1

#old way of setting initial values
# for i in range(arraysize[0]):
#     for j in range(arraysize[1]):
        
#         #DOES NOT WORK
#         #use random triangular to get distribution weighted around 0 
        # x[i,j] = random.triangular(pos_int_bounds[0], 0, pos_int_bounds[1])
        # y[i,j] = random.triangular(pos_int_bounds[0], 0, pos_int_bounds[1])
        # vx[i,j] = random.triangular(vel_ints_bounds[0], 0, vel_ints_bounds[1])
        # vy[i,j] = random.triangular(vel_ints_bounds[0], 0, vel_ints_bounds[1])

#         #Don't know the distributions of charge and k, so just doing uniform generation
#         charge_matrix[i,j] = rng.integers(0,charge)   #assume all have negative charge
#         #spring constants are actually k/m
#         kx_matrix[i,j] = freq_to_k(rng.integers(frange[0],frange[1]))   # in 1/s^2
#         ky_matrix[i,j] = freq_to_k(rng.integers(frange[0],frange[1]))   # in 1/s^2
    
print(x)
print(vx)
print(charge_matrix)
print(fx_matrix)
print(fy_matrix)


sep = [50,60,70,80,100]          # separation in um
mpl.rcParams.update({'font.size': 18})

if arraysize[0] * arraysize[1] > 2:
    figa, axa = plt.subplots(1, len(sep))
    cax = figa.add_axes(rect=(0.2,0.2,0.6,0.03))
    #figa.suptitle('Correlation of Motion of a 5x5 Array of Spheres')
    
    #figb, axb = plt.subplots(2,1, sharex=True)

else:
    jointfig = plt.figure()
    grid = plt.GridSpec(11, 10, wspace=3.5, hspace=4)
    sph0 = jointfig.add_subplot(grid[:4, :7])
    sph1 = jointfig.add_subplot(grid[5:9, :7])
    jointcor = jointfig.add_subplot(grid[:, 7:])
    sphraxes = [sph0, sph1]
    jointcor.tick_params(labelsize=14)
    sph0.tick_params(labelsize=14)
    sph1.tick_params(labelsize=14)
    jointcor.set_title('Correlation vs Separation',fontsize = 26)
    jointcor.set_xlabel(r'Separation ($\mu m$)',fontsize=20)
    jointcor.set_ylabel('Pearson Correlation Coefficient',fontsize=20)
    
    colorvalue1 = np.linspace(0,0.8,len(sep))
    colorvalue2 = np.linspace(0.4,1,len(sep))
    viridis = mpl.colormaps['viridis'].resampled(20)
    inferno = mpl.colormaps['inferno'].resampled(20)
    colorcodes1 = [viridis(colorvalue1[i]) for i in range(len(sep))]
    colorcodes2 = [inferno(colorvalue2[i]) for i in range(len(sep))]
    figs = {}
    axs = {}

k = 0
for d in sep:
    
    #xsaves, etc. do not have the right number of samples in them
    save_times, xsaves, ysaves, vxsaves, vysaves  = baoab(arraysize, timespan, dt, fs, gamma, kBT,\
                                                                    x, y, vx, vy, kx_matrix, ky_matrix,\
                                                                    charge_matrix, charge_const, d*10**-6, startrec)
    segmentsize = round(fs/5)
    fftbinning = 1024
    first = True
    for i in range(arraysize[0]):
        for j in range(arraysize[1]):
            if first == True:
                xarray = np.array(xsaves[j][i]).reshape(-1,1)
                yarray = np.array(ysaves[j][i]).reshape(-1,1)
                vxarray = np.array(vxsaves[j][i]).reshape(-1,1)
                vyarray = np.array(vysaves[j][i]).reshape(-1,1)
                
                freqx, PSDx = welch(xarray, fs, 'hann', segmentsize, segmentsize/2, fftbinning, 'constant', True, 'density', 0,'mean')
                freqy, PSDy = welch(yarray, fs, 'hann', segmentsize, segmentsize/2, fftbinning, 'constant', True, 'density', 0,'mean')

                xPSDarray = PSDx
                yPSDarray = PSDy
                first = False
            else:
                addedx = np.array(xsaves[j][i]).reshape(-1,1)
                addedy = np.array(ysaves[j][i]).reshape(-1,1)

                freqx, PSDx = welch(addedx, fs, 'hann', segmentsize, segmentsize/2, fftbinning, 'constant', True, 'density', 0,'mean')
                freqy, PSDy = welch(addedy, fs, 'hann', segmentsize, segmentsize/2, fftbinning, 'constant', True, 'density', 0,'mean')

                xPSDarray = np.concatenate((xPSDarray, PSDx),axis=1)
                yPSDarray = np.concatenate((yPSDarray, PSDy),axis=1)

                xarray = np.concatenate((xarray, addedx),axis=1)
                yarray = np.concatenate((yarray, addedy),axis=1)
                vxarray = np.concatenate((vxarray, np.array(vxsaves[j][i]).reshape(-1,1)),axis=1)
                vyarray = np.concatenate((vyarray, np.array(vysaves[j][i]).reshape(-1,1)),axis=1)  


    print(xarray.shape)
    xcorrmatrix = np.corrcoef(xarray,rowvar=False)
    ycorrmatrix = np.corrcoef(yarray,rowvar=False)
    print(xcorrmatrix)
    print(ycorrmatrix)

    if arraysize[0] * arraysize[1] > 2:

        for l in range(xcorrmatrix.shape[0]):
            for m in range(xcorrmatrix.shape[1]):

                if xcorrmatrix[m][l] == 1:
                    xcorrmatrix[m][l] = 0
                if ycorrmatrix[m][l] == 1:
                    ycorrmatrix[m][l] = 0



        spherenames = [str(x+1) for x in range(arraysize[0]*arraysize[1])]
        norm = LogNorm()
        if k == len(sep) - 1:
            plot_cbar = True
            cbar_kws = {'shrink' : 0.8,
                        'orientation': 'horizontal'}
            cbar_ax = cax
            cax.set_title('Correlation Coefficients',fontsize=26)
            
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
        sn.heatmap(symcor, mask=diagmask, square=True, cmap = 'viridis', vmin=-0.2, vmax=0.2, ax=axa[k], cbar=plot_cbar, cbar_ax = cbar_ax, cbar_kws=cbar_kws)
        axa[k].tick_params(axis='both', which='major', labelsize=18)
        #ax[i].set_xticks(np.arange(xcor.shape[1])+.5, labels=spherenames,fontsize=16)
        #ax[i].set_yticks(np.arange(xcor.shape[0])+.5, labels=spherenames,fontsize=16)
        #plt.setp(ax[i].get_xticklabels(), rotation=90)
        axa[k].set_title(str(int(d)) + r'$~\mu$m Spacing', fontsize=26, pad=15)
        axa[k].set_xlabel('Sphere Index', fontsize=22, labelpad=5)
        axa[k].set_ylabel('Sphere Index',fontsize=22,labelpad=5)

    else:

        coherfreq = coherence(xarray[:,0],xarray[:,1], fs = fs, window='hann', nperseg=segmentsize, nfft=fftbinning)[0]
        xcohermatrix = coherence(xarray[:,0],xarray[:,1], fs, nperseg=segmentsize, nfft=fftbinning)[1]
        ycohermatrix = coherence(yarray[:,0],yarray[:,1], fs, nperseg=segmentsize, nfft=fftbinning)[1]

        xcorrvalue = 0
        for z in range(len(coherfreq)):
            if coherfreq[z] > 80:
                xcorrvalue = xcorrvalue + xcohermatrix[z]
        
        ycorrvalue = 0
        for z in range(len(coherfreq)):
            if coherfreq[z] > 80:
                ycorrvalue = ycorrvalue + ycohermatrix[z]

        figs[k], axs[k] = plt.subplots()
        axs[k].semilogy(coherfreq, xcohermatrix, label = 'X')
        axs[k].semilogy(coherfreq, ycohermatrix, label = 'Y')
        axs[k].legend()
        axs[k].set_title('%d separation Coherence' % d)

        jointcor.scatter(d, abs(xcorrvalue), marker='s' ,color = '#1E88E5', label='X Motion')
        jointcor.scatter(d, abs(ycorrvalue), color = '#004D40', label='Y Motion')
        for j in range(xPSDarray.shape[1]):
            label_name = str(int(d)) + r' $\mu$m'

            sphraxes[j].plot(freqx, np.sqrt(xPSDarray[:,j]), color = colorcodes1[k], label=label_name)
            sphraxes[j].plot(freqy, np.sqrt(yPSDarray[:,j]), color = colorcodes2[k], label=label_name)
        sph0.set_xlim([5,250])
        sph0.set_xlabel('Frequency (Hz)', fontsize = 20)
        sph0.set_ylabel(r'ASD ($m/ \sqrt{Hz}$)', fontsize = 20)
        sph0.set_title('Sphere 0', fontsize = 26)

        sph1.set_xlim([5,250])
        sph1.set_xlabel('Frequency (Hz)', fontsize = 20)
        sph1.set_ylabel(r'ASD ($m/ \sqrt{Hz}$)', fontsize = 20)
        sph1.set_title('Sphere 1', fontsize = 26)

        h, l = sph1.get_legend_handles_labels()
        ph = [plt.plot([],marker="", ls="")[0]]*2
        handles = ph + h
        labels = ['X Data:', "Y Data:"] + l
        leg = sph1.legend(handles, labels, fontsize=12, ncols=len(labels)/2, loc="upper left", bbox_to_anchor=(-0.1, -0.3), borderaxespad=0.1)
        for vpack in leg._legend_handle_box.get_children()[:1]:
            for hpack in vpack.get_children():
                hpack.get_children()[0].set_width(0)

    k+=1

    


plt.show()
