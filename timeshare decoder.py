# -*- coding: utf-8 -*-
"""
Looking at data from qpd
"""

import pandas as pd
import numpy as np
import scipy as sp
import math
import matplotlib.pyplot as plt
import os
from scipy import fft
from scipy import special
from scipy.signal.windows import blackman
from scipy.signal import find_peaks


##############################################################################

#variables to define for each file

path = r"C:\Users\Ben\Documents\Research\Moore Lab\timeshare qpd data"
os.chdir(path)

filename = '2sp8_28.csv'

inc = 2E-7
numbeams = 4 #4 beams total
timeperbeam = 1E-4
startbeamnum = 2 #beam number indexes from 0
totaldelaytime = 4.8E-5


##############################################################################



f = h5py.File(filename, 'r')
group = f['beads/data/pos_data']
samplingrate = group.attrs['Fsamp']




df = pd.read_csv(filename, delimiter=',', header=2, usecols=[0,1,2,3,4], names=['Time','CH1','CH2','CH3','trig'])
df.Time = df.Time*inc
triglocs = []
i = 5
while i < len(df.Time):
    if df.trig[i] < (df.trig[i-5]-0.03) or (df.trig[i-5]+0.03) < df.trig[i]: #this is for the scope being at 10x setting
        triglocs.append(i)
        i = i+10
    else:
        i = i+1
   

fig, ax = plt.subplots()
ax.plot(df.Time,df.CH1)
ax.plot(df.Time,df.CH2)
ax.plot(df.Time,df.CH3)
ax.plot(df.Time,df.trig)
#print(df['Time'].iloc[-1])

fig, ax2 = plt.subplots()
ax2.set_xlim(0,2E3)
ax2.plot(df.Time*1E6,df.CH1)
ax2.plot(df.Time*1E6,df.CH2)
ax2.plot(df.Time*1E6,df.CH3)
#ax2.plot(df.Time*1E6,df.trig)
ax2.vlines(df.Time[triglocs]*1E6,-1, 1, colors = 'r', linestyles = 'dashed', linewidths = 0.5)
ax2.legend(['QPD X','QPD Y', 'QPD Sum'],loc = 'upper right',framealpha = 1)
ax2.set_xlabel('Time [us]')
ax2.set_ylabel('QPD Response [V]')
ax2.grid()


indexdelay = int(totaldelaytime/inc)

time = [ [] for _ in range(numbeams)]
x = [ [] for _ in range(numbeams)]
y = [ [] for _ in range(numbeams)]
z = [ [] for _ in range(numbeams)]

sniplength = []
minsnip = 258
xsnipfft = [ np.array([]) for _ in range(numbeams)]
ysnipfft = [ np.array([]) for _ in range(numbeams)]
zsnipfft = [ np.array([]) for _ in range(numbeams)]

snipfreqs = fft.rfftfreq(minsnip, inc)

j = startbeamnum
for i in range(len(triglocs)):
    j=(j+1)%numbeams
    if i+1 >= len(triglocs):
        print(j)
    else:
        xsniplist = np.array(df.CH1[triglocs[i]+indexdelay:triglocs[i+1]])
        ysniplist = np.array(df.CH2[triglocs[i]+indexdelay:triglocs[i+1]])
        zsniplist = np.array(df.CH3[triglocs[i]+indexdelay:triglocs[i+1]])
        xsnip = np.mean(xsniplist)
        ysnip = np.mean(ysniplist)
        zsnip = np.mean(zsniplist)
        
        sniplength.append(len(df.CH1[triglocs[i]+indexdelay:triglocs[i+1]]))
        
        xloopfft = np.array(2 * inc/minsnip * np.abs(fft.rfft(xsniplist[:minsnip])**2))
        yloopfft = np.array(2 * inc/minsnip * np.abs(fft.rfft(ysniplist[:minsnip])**2))
        zloopfft = np.array(2 * inc/minsnip * np.abs(fft.rfft(zsniplist[:minsnip])**2))
        if xsnipfft[j].size == 0:
            xsnipfft[j] = xloopfft
            ysnipfft[j] = yloopfft
            zsnipfft[j] = zloopfft 
        else:
            xsnipfft[j] = np.add(xsnipfft[j], xloopfft)
            ysnipfft[j] = np.add(ysnipfft[j], yloopfft)
            zsnipfft[j] = np.add(zsnipfft[j], zloopfft)
            
        time[j].append(df.Time[triglocs[i]])
        x[j].append(xsnip)
        y[j].append(ysnip)
        z[j].append(zsnip)
    

fig, axa = plt.subplots()
fig, axb = plt.subplots()
fig, axc = plt.subplots() 

fig, ax3 = plt.subplots()
fig, ax4 = plt.subplots()
fig, ax5 = plt.subplots()

legend_dir= []
for i in range(0,numbeams):
    
    ax3.plot(time[i],x[i])
    ax4.plot(time[i],y[i])
    ax5.plot(time[i],z[i])
    legend_dir.append('Trap '+ str(i+1))
    
    xsnipfft[i] = xsnipfft[i]/len(time[0])
    ysnipfft[i] = ysnipfft[i]/len(time[0])
    zsnipfft[i] = zsnipfft[i]/len(time[0])
    axa.loglog(snipfreqs, xsnipfft[i])
    axb.loglog(snipfreqs, ysnipfft[i])
    axc.loglog(snipfreqs, zsnipfft[i])
    
ax3.set_xlabel('Time [s]')
ax3.set_ylabel('Signal X Channel [V]')
ax3.legend(legend_dir)
plt.grid()


ax4.set_xlabel('Time [s]')
ax4.set_ylabel('Signal Y Channel [V]')
ax4.legend(legend_dir)

ax5.set_xlabel('Time [s]')
ax5.set_ylabel('Signal SUM Channel [V]')
ax5.legend(legend_dir)


axa.set_xlabel('Frequency [Hz]')
axa.set_ylabel('PSD X [V^2/Hz]')
axa.set_title('Average of PSD for beam illumination time')
axa.legend(legend_dir)

axb.set_xlabel('Frequency [Hz]')
axb.set_ylabel('PSD Y [V^2/Hz]')
axb.set_title('Average of PSD for beam illumination time')
axb.legend(legend_dir)

axc.set_xlabel('Frequency [Hz]')
axc.set_ylabel('PSD Z [V^2/Hz]')
axc.set_title('Average of PSD for beam illumination time')
axc.legend(legend_dir)

timeshareinc = timeperbeam*numbeams

#coerce all lists to be same length so can process fft without issues
for i in range(numbeams):
    x[i] = x[i][:(min(len(k) for k in x))]
    y[i] = y[i][:(min(len(k) for k in y))]
    z[i] = z[i][:(min(len(k) for k in z))]
    time[i] = time[i][:(min(len(k) for k in time))]

timesharesize = len(time[0])
w = blackman(timesharesize)


fftx = [ [] for _ in range(numbeams)]
ffty = [ [] for _ in range(numbeams)]
fftz = [ [] for _ in range(numbeams)]
freqs = fft.rfftfreq(timesharesize, timeshareinc)



fig, ax6 = plt.subplots()
plt.grid()

fig, ax7 = plt.subplots()
plt.grid()

fig, ax8 = plt.subplots()
plt.grid()


legend_dir= []
for i in range(0,numbeams):
    fftx[i] = 2 * timeshareinc/timesharesize * np.abs(fft.rfft(x[i])**2)
    ffty[i] = 2 * timeshareinc/timesharesize * np.abs(fft.rfft(y[i])**2)
    fftz[i] = 2 * timeshareinc/timesharesize * np.abs(fft.rfft(z[i])**2)
    
    ax6.loglog(freqs,fftx[i])
    ax7.loglog(freqs,ffty[i])
    ax8.loglog(freqs,fftz[i])
    legend_dir.append('Trap '+ str(i+1))

ax6.set_xlabel('Frequency [Hz]')
ax6.set_ylabel('PSD X [V^2/Hz]')
ax6.legend(legend_dir)

ax7.set_xlabel('Frequency [Hz]')
ax7.set_ylabel('PSD Y[V^2/Hz]')
ax7.legend(legend_dir)

ax8.set_xlabel('Frequency [Hz]')
ax8.set_ylabel('PSD Z[V^2/Hz]')
ax8.legend(legend_dir)




# peaks1, _ = find_peaks(beam1fft,threshold=2E-8)
# peaks2, _ = find_peaks(beam2fft,threshold=2E-8)

# fig, ax6 = plt.subplots()
# ax6.loglog(x,beam1fft)
# ax6.loglog(x,beam2fft)
# ax6.loglog(x,beam3fft)
# ax6.loglog(x[:len(beam4fft)],beam4fft)
# ax6.loglog(x[peaks1],beam1fft[peaks1], 'x')
# for i, txt in enumerate(np.around(x[peaks1])):
#     ax6.annotate(txt, (x[peaks1[i]],beam1fft[peaks1[i]]))
# ax6.loglog(x[peaks2],beam1fft[peaks2], 'x')
# for i, txt in enumerate(np.around(x[peaks2])):
#     ax6.annotate(txt, (x[peaks2[i]],beam1fft[peaks2[i]]))
# ax6.set_xlabel('Frequency [Hz]')
# ax6.set_ylabel('PSD X [V^2/Hz]')
# plt.grid()



# peaks1, _ = find_peaks(beam1yfft,threshold=2E-8)

# fig, ax7 = plt.subplots()
# ax7.loglog(x,beam1yfft)
# ax7.loglog(x,beam2yfft)
# ax7.loglog(x,beam3yfft)
# ax7.loglog(x[:len(beam4fft)],beam4yfft)
# ax7.loglog(x[peaks1],beam1yfft[peaks1], 'x')
# for i, txt in enumerate(np.around(x[peaks1])):
#     ax7.annotate(txt, (x[peaks1[i]],beam1yfft[peaks1[i]]))
# ax7.set_xlabel('Frequency [Hz]')
# ax7.set_ylabel('PSD Y[V^2/Hz]')
# plt.grid()


# peaks1, _ = find_peaks(beam2zfft,threshold=7E-8)



# ax8.loglog(x,beam1zfft)
# ax8.loglog(x,beam2zfft)
# ax8.loglog(x,beam3zfft)
# ax8.loglog(x[:len(beam4fft)],beam4zfft)
# #ax8.loglog(x[peaks1],beam2zfft[peaks1], 'x')
# #for i, txt in enumerate(np.around(x[peaks1])):
# #    ax8.annotate(txt, (x[peaks1[i]],beam2zfft[peaks1[i]]))
# ax8.set_xlabel('Frequency [Hz]')
# ax8.set_ylabel('PSD Z[V^2/Hz]')
# plt.grid()
