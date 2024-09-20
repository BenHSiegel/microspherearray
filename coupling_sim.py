'''
BAOAB coupled langevin estimator
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch


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

def baoab(motion_eq, timespan, dt, fs, gamma, kBT, pos_init1, pos_init2, vel_init1, vel_init2, k1, k2, CC, sep):
    
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
        force1 = full_motion_eq(x1,x2,k1,-CC, sep)
        force2 = full_motion_eq(x2,x1,k2,CC, sep)
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
        force1 = full_motion_eq(x1,x2,k1,-CC, sep)
        force2 = full_motion_eq(x2,x1,k2,CC, sep)
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

timespan = 100
dt = 0.0001
fs = 1000

pos_ints = [3e-7,-2e-7]     #starting position in m
vel_ints = [0,0]

pressure = 0.4      # in mbar
temp = 295          # in K
kBT = 4.073e-21     # for T = 295K (in N m)
gamma = 9.863e-10 * pressure / np.sqrt(temp)     #Epstein drag using 10um sphere (in kg/s)

f1 = 160            # in Hz
f2 = 175            # in Hz
k1 = freq_to_k(f1)  # in N m^-1 kg^-1
k2 = freq_to_k(f2)  # in N m^-1 kg^-1

sep = [210, 180, 140, 120, 100, 70, 60, 50, 40, 30]          # separation in um
figs = {}
axs = {}
i = 0
for d in sep:
    charge = 500       # number of electrons
    charge_const = 2.30708e-16 * charge**2      #Q^2 / (4 pi epsilon_0 * 1ng) in N m^2 / kg
    
    #approx_charge_coupling = 230.708 * charge**2 / (d)**3 # in N m^-1 kg^-1

    times, positions1, velocities1, positions2, velocities2  = baoab(motion_eq, timespan, dt, fs, gamma, kBT,\
                                                                        pos_init1=pos_ints[0], pos_init2=pos_ints[1],\
                                                                        vel_init1=vel_ints[0], vel_init2=vel_ints[1],\
                                                                        k1=k1, k2=k2, CC=charge_const, sep=(d*10**-6))


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
