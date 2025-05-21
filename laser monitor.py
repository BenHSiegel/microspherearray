import h5py
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

filename = r'C:\Users\yalem\Desktop\laser fluctuations.h5'
hf = h5py.File(filename, 'r')

setpoint = hf.attrs['Set point (V)']
print(setpoint)
data = hf.get('Slow beam monitor')
time = data[:,0]
mean = data[:,1]
std = data[:,2]

#plt.plot(time/3600, (mean/setpoint - 1)*100, label='Mean')
plt.plot(time, std/mean, label='Std')
plt.xlabel('Time (h)')
plt.ylabel(r'Coefficient of variation of each 0.3s chunk')
plt.title('Laser Fluctuations')
#plt.legend()
plt.show()