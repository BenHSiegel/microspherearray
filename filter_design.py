'''
FIR tap design
'''

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy.signal import minimum_phase, freqz, group_delay



fig,ax = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()

fs = 12048
fc = 50
for numtaps in range(1,6):
    coefs = signal.firwin(numtaps, fc,fs=fs,window='boxcar')
    coefstr = np.round(coefs,4)
    print(coefs)
    w, h = signal.freqz(b=coefs, a=1,worN=1024)
    wtr, htr = signal.freqz(b=coefstr, a=1,worN=1024)
    x = w * fs * 1.0 / (2 * np.pi)
    phase = -1* numtaps/2 * w / (2 * np.pi)
    y = 20 * np.log10(abs(h))
    ytr = 20 * np.log10(abs(htr))
    ax.plot(x, y,label=numtaps)
    #ax.plot(x,abs(htr),label="Truncated +" + str(numtaps))
    #ax3.plot(x,abs(h/htr),label=numtaps)
    #ax2.plot(x,phase,label=numtaps)

ax.set_ylabel('Amplitude [dB]')
ax.set_xlabel('Frequency [Hz]')
ax.set_title('Frequency response')
ax.legend()
ax.grid(which='both', linestyle='-', color='grey')

# ax3.set_ylabel('Difference with rounding')
# ax3.set_xlabel('Frequency [Hz]')
# ax3.set_title('Frequency response')
# ax3.legend()
# ax3.grid(which='both', linestyle='-', color='grey')


# ax2.set_ylabel('Phase')
# ax2.set_xlabel('Frequency')
# ax2.set_title('Frequency response')
# ax2.legend()
# ax2.grid(which='both', linestyle='-', color='grey')

plt.show()

# coefs = [0.19201,0.234355,0.234355,0.19201]
# fs = 2500
# w, h = signal.freqz(b=coefs, a=1,worN=1024)
# x = w * fs * 1.0 / (2 * np.pi)
# y = 20 * np.log10(abs(h))
# plt.plot(x,y)
# plt.show()