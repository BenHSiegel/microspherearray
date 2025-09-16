import numpy as np
from scipy.signal import iirpeak, freqz
import matplotlib.pyplot as plt

# Set global font to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

'''
# Define filter parameters
fs = 50000.0  # Sampling frequency in Hz
f0 = 100.0     # Frequency to be targeted by the peak filter in Hz
Q = 30.0      # Quality factor

# Design the IIR peak filter
b, a = iirpeak(f0 / (fs / 2), Q)

# Quantize coefficients to fixed-point representation
def quantize(coefficients, num_bits=16):
    scale = 2 ** (num_bits - 1)  # Scale factor for fixed-point
    quantized = np.round(coefficients * scale) / scale
    return quantized

b_quantized = quantize(b)
a_quantized = quantize(a)

print("Original coefficients:")
print("b:", b)
print("a:", a)
print("\nQuantized coefficients:")
print("b_quantized:", b_quantized)
print("a_quantized:", a_quantized)

# Frequency response of quantized filter
w_quantized, h_quantized = freqz(b_quantized, a_quantized, worN=8000)
frequencies = w_quantized * fs / (2 * np.pi)

# Frequency response of unquantized filter
w_unquantized, h_unquantized = freqz(b, a, worN=8000)

# Zoom range around the targeted frequency
zoom_range = (0, 200)

# Combined plot with dual y-axes
fig, ax1 = plt.subplots()

# Plot magnitude response on the left y-axis
ax1.plot(frequencies, 20 * np.log10(abs(h_quantized)), 'b', label='Quantized')
ax1.plot(frequencies, 20 * np.log10(abs(h_unquantized)), 'b--', label='Unquantized')
ax1.set_xlabel('Frequency (Hz)')
ax1.set_ylabel('Amplitude (dB)', color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax1.grid()
ax1.set_xlim(zoom_range)
ax1.set_ylim(-60, 5)  # Set y-limits for amplitude response

# Create a secondary y-axis for the phase response
ax2 = ax1.twinx()
ax2.plot(frequencies, np.angle(h_quantized), 'r', label='Quantized Phase (radians)')
ax2.plot(frequencies, np.angle(h_unquantized), 'r--', label='Unquantized Phase (radians)')
ax2.set_ylabel('Phase (radians)', color='r')
ax2.tick_params(axis='y', labelcolor='r')

# Add a title and legend
plt.title('IIR Peak Filter Frequency and Phase Response (Quantized vs Unquantized)')
ax1.legend(loc="upper right")
# Show the plot
plt.show()
'''

# Define Lorentzian function
def lorentzian(f, f0, gamma):
    return 1/((f0**2 - f**2)**2 + f**2 * (gamma)**2)

# Parameters for the Lorentzian
f0_lorentzian = 100.0  # Resonance frequency in Hz
gamma_values = [0.001, 0.01, 0.1, 1.0]  # Linewidth values
frequencies_lorentzian = np.linspace(60, 140, 3000)  # Frequency range for plotting

plt.figure()

# Plot the Lorentzian amplitude for each gamma value (in dB)
for gamma_lorentzian in gamma_values:
    amplitude_lorentzian = np.sqrt(lorentzian(frequencies_lorentzian, f0_lorentzian, gamma_lorentzian))
    amplitude_lorentzian_lin = (amplitude_lorentzian / np.max(amplitude_lorentzian))
    amplitude_lorentzian_db = 10 * np.log10(amplitude_lorentzian / np.max(amplitude_lorentzian))  # Convert to dB and normalize
    plt.semilogy(frequencies_lorentzian, amplitude_lorentzian_lin, label=f'$\Gamma_0$ = {gamma_lorentzian} Hz')
plt.xlim(85, 115)
plt.xlabel('Frequency (Hz)', fontsize=14)
plt.ylabel('Normalized Amplitude', fontsize=14)
plt.title('Simulated Particle Response for 100 Hz Resonance', fontsize=18)
plt.legend(fontsize=12)
plt.grid()
plt.show()

plt.figure()

for gamma_lorentzian in gamma_values:
    # Calculate Lorentzian amplitude
    amplitude_lorentzian = lorentzian(frequencies_lorentzian, f0_lorentzian, gamma_lorentzian)
    amplitude_peak = lorentzian(f0_lorentzian, f0_lorentzian, gamma_lorentzian)
    # Plot the response ratio from the peak frequency to 4 times gamma_lorentzian
    frequency_range = np.linspace(f0_lorentzian - 40, f0_lorentzian + 40, 1000)
    amplitude_range = lorentzian(frequency_range, f0_lorentzian, gamma_lorentzian)
    response_ratios = 10 * np.log10(amplitude_peak / amplitude_range)  # Convert to dB

    plt.plot(frequency_range - 100, response_ratios, label=f'$\Gamma_0$ = {gamma_lorentzian} Hz')

plt.xlabel('Frequency Offset f - f0 (Hz)')
plt.ylabel('Amplitude Suppression (dB)')
plt.title('Response Ratio')
plt.legend()
plt.grid()
plt.show()
# Additional figure with f0_lorentzian = 250
f0_lorentzian = 250.0  # Update resonance frequency

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

# First plot: f0_lorentzian = 100
response_ratios_100 = []
for gamma_lorentzian in gamma_values:
    amplitude_lorentzian = lorentzian(frequencies_lorentzian, 100.0, gamma_lorentzian)
    amplitude_peak = lorentzian(100.0, 100.0, gamma_lorentzian)
    frequency_range = np.linspace(100.0 - 40, 100.0 + 40, 1000)
    amplitude_range = lorentzian(frequency_range, 100.0, gamma_lorentzian)
    response_ratios = 10 * np.log10(amplitude_peak / amplitude_range)
    response_ratios_100.append(response_ratios)
    ax1.plot(frequency_range - 100, response_ratios, label=f'$\gamma$ = {gamma_lorentzian} Hz')

ax1.set_xlabel('Frequency Offset f - f0 (Hz)')
ax1.set_ylabel('Amplitude Suppression (dB)')
ax1.set_title('Response Ratio (f0 = 100 Hz)')
ax1.legend()
ax1.grid()

# Second plot: f0_lorentzian = 250
response_ratios_250 = []
for gamma_lorentzian in gamma_values:
    amplitude_lorentzian = lorentzian(frequencies_lorentzian, f0_lorentzian, gamma_lorentzian)
    amplitude_peak = lorentzian(f0_lorentzian, f0_lorentzian, gamma_lorentzian)
    frequency_range = np.linspace(f0_lorentzian - 40, f0_lorentzian + 40, 1000)
    amplitude_range = lorentzian(frequency_range, f0_lorentzian, gamma_lorentzian)
    response_ratios = 10 * np.log10(amplitude_peak / amplitude_range)
    response_ratios_250.append(response_ratios)
    ax2.plot(frequency_range - 250, response_ratios, label=f'$\gamma$ = {gamma_lorentzian} Hz')

ax2.set_xlabel('Frequency Offset f - f0 (Hz)')
ax2.set_ylabel('Amplitude Suppression (dB)')
ax2.set_title('Response Ratio (f0 = 250 Hz)')
ax2.legend()
ax2.grid()

# Third plot: Difference between the two
for i, gamma_lorentzian in enumerate(gamma_values):
    difference = response_ratios_100[i] - response_ratios_250[i]
    ax3.plot(frequency_range - 100, difference, label=f'$\gamma$ = {gamma_lorentzian} Hz')

ax3.set_xlabel('Frequency Offset f - f0 (Hz)')
ax3.set_ylabel('Difference in Suppression (dB)')
ax3.set_title('Difference Between f0 = 100 Hz and f0 = 250 Hz')
ax3.legend()
ax3.grid()

plt.tight_layout()
plt.show()

# Plot the response ratio for a Lorentzian with gamma = 2 for f0_lorentzian from 60 to 120
gamma_lorentzian = 2
f0_lorentzian_values = np.linspace(60, 120, 7)  # Resonance frequencies from 60 Hz to 120 Hz
frequency_range = np.linspace(0, 200, 1000)  # Extended frequency range for plotting

plt.figure()

# Calculate the response ratios for the minimum and maximum f0_lorentzian values
response_ratios_min = []
response_ratios_max = []

for frequency in frequency_range:
    amplitude_peak_min = lorentzian(60, 60, gamma_lorentzian)
    amplitude_range_min = lorentzian(frequency, 60, gamma_lorentzian)
    response_ratios_min.append(10 * np.log10(amplitude_peak_min / amplitude_range_min))

    amplitude_peak_max = lorentzian(120, 120, gamma_lorentzian)
    amplitude_range_max = lorentzian(frequency, 120, gamma_lorentzian)
    response_ratios_max.append(10 * np.log10(amplitude_peak_max / amplitude_range_max))

# Convert lists to numpy arrays for plotting
response_ratios_min = np.array(response_ratios_min)
response_ratios_max = np.array(response_ratios_max)

# Normalize the x-axis separately for the 60 Hz and 120 Hz cases
normalized_frequency_min = frequency_range - 60
normalized_frequency_max = frequency_range - 120

# Filter data to only include points where normalized frequency is between -60 and 60
mask_min = (normalized_frequency_min >= -60) & (normalized_frequency_min <= 60)
mask_max = (normalized_frequency_max >= -60) & (normalized_frequency_max <= 60)

normalized_frequency_min = normalized_frequency_min[mask_min]
response_ratios_min = response_ratios_min[mask_min]

normalized_frequency_max = normalized_frequency_max[mask_max]
response_ratios_max = response_ratios_max[mask_max]

# Plot the f0 = 60 Hz case
plt.plot(normalized_frequency_min, response_ratios_min, label='$f_0$ = 60 Hz', color='blue', linestyle='--')

# Plot the f0 = 120 Hz case
plt.plot(normalized_frequency_max, response_ratios_max, label='$f_0$ = 120 Hz', color='blue', linestyle=':')

# Set x-axis limits to range from -60 to 60
plt.xlim(-60, 60)

# Fill the area between the two curves
plt.fill_between(
    normalized_frequency_min, 
    response_ratios_min, 
    response_ratios_max,
    color='blue', 
    alpha=0.3
)

plt.xlabel('Frequency Offset From Resonance (Hz)')
plt.ylabel('Amplitude Suppression (dB)')
plt.title('Response Ratio for Particles \n in typical resonance range and $\Gamma$ = 2 Hz')
plt.legend(loc='lower right')
plt.grid()
plt.show()