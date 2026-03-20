# Libraries
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import special
from scipy.signal import find_peaks
import os
import re
from collections import defaultdict
from sklearn.metrics import r2_score


def read_pulsed_csv(folder, min_peak_distance=50, min_peak_prominence=None, plot=False):
    """
    Read all diodexx_knife_in_yy_zz.csv files from a folder, group by knife position
    yy, and compute the average power at each position from the local maxima.

    Each CSV is expected to be a single row of comma-separated power values (no header).

    Parameters
    ----------
    folder              : path to folder containing the CSV files
    min_peak_distance   : minimum sample separation between peaks passed to
                          scipy.signal.find_peaks (default 50)
    min_peak_prominence : minimum peak prominence; None uses 5% of the signal
                          range for each file
    plot                : if True, plot raw data + detected peaks for every file

    Returns
    -------
    yy_positions   : sorted list of knife-edge positions (yy values)
    avg_power      : corresponding list of average power at each position
    per_file_avgs  : dict mapping yy -> list of per-file peak averages (for inspection)
    """
    pattern = re.compile(r'^diode(\d+)_knife_in_(\d+)_(\d+)\.csv$')

    # Collect per-file peak averages keyed by yy
    groups = defaultdict(list)

    for filename in sorted(os.listdir(folder)):
        m = pattern.match(filename)
        if not m:
            continue
        xx, yy, zz = int(m.group(1)), int(m.group(2)), int(m.group(3))

        filepath = os.path.join(folder, filename)
        with open(filepath, 'r') as f:
            raw = f.read().split(',')
        data = np.array([float(v) for v in raw if v.strip()])

        # Determine prominence threshold if not supplied
        prominence = min_peak_prominence
        if prominence is None:
            prominence = 0.05 * (data.max() - data.min())

        peaks, props = find_peaks(data,
                                  distance=min_peak_distance,
                                  prominence=prominence)

        if len(peaks) == 0:
            print(f'  WARNING: no peaks found in {filename}, skipping.')
            continue

        file_avg = data[peaks].mean()
        groups[yy].append(file_avg)

        print(f'{filename}  ->  {len(peaks)} peaks,  mean peak power = {file_avg:.4f}')

        if plot:
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(data, lw=0.5, label='Data')
            ax.plot(peaks, data[peaks], 'x', color='red', label='Peaks')
            ax.set_title(filename)
            ax.set_xlabel('Sample index')
            ax.set_ylabel('Power')
            ax.legend()
            plt.tight_layout()
            plt.show()

    # Average across files sharing the same yy
    yy_positions = sorted(groups.keys())
    avg_power = [np.mean(groups[yy]) for yy in yy_positions]

    print('\n--- Average power per knife position ---')
    for yy, p in zip(yy_positions, avg_power):
        print(f'  yy = {yy:3d}  ->  avg power = {p:.4f}')

    return yy_positions, avg_power, dict(groups)


def load_data(path, delimiter='\t', header=32, time_col=0, power_col=1):
    """Load all .txt knife-edge data files from a directory."""
    os.chdir(path)
    file_names = []
    data_frames = []
    for filename in sorted(os.listdir(path)):
        if filename.endswith(".txt"):
            print(filename)
            file_names.append(filename)
            df = pd.read_csv(filename, delimiter=delimiter, header=header,
                             usecols=[time_col, power_col], names=['Time', 'Power'])
            data_frames.append(df)
    return file_names, data_frames


def erf_model(x, x0, p_max, w, offset):
    """Error function model for knife-edge beam profile (power increases with position)."""
    return offset + (1/2) * p_max * (1 + special.erf(np.sqrt(2) * (x - x0) / w))


def erf_model_flipped(x, x0, p_max, w, offset):
    """Flipped error function model (power decreases with position)."""
    return offset + (1/2) * p_max * (1 - special.erf(np.sqrt(2) * (x - x0) / w))


def fit_erf_anchored(x, y, n_flat=3, flipped=None):
    """
    Fit a knife-edge ERF with amplitude and offset anchored to the endpoint averages.

    When only part of the ERF is captured (e.g. the bottom flat is not reached),
    a 4-parameter free fit struggles because p_max and offset are unconstrained.
    This function anchors the top and bottom asymptotes to the mean of the first/last
    n_flat sorted data points, then does a fast 2-parameter fit for (x0, w) only.

    Parameters
    ----------
    x, y    : knife-edge position and power arrays (any order)
    n_flat  : number of endpoint samples to average for asymptote anchoring (default 3)
    flipped : True = descending ERF (knife cuts in), False = ascending, None = auto

    Returns
    -------
    x0_fit, w_fit, y_top, y_bottom, r2, model_fn
        model_fn(x_array) -> y_array using the fitted parameters
    """
    sort_idx = np.argsort(x)
    xs, ys = x[sort_idx], y[sort_idx]

    n_flat = min(n_flat, len(xs) // 2)
    y_start = np.mean(ys[:n_flat])   # average of the first n_flat points
    y_end   = np.mean(ys[-n_flat:])  # average of the last n_flat points

    if flipped is None:
        flipped = y_start > y_end

    if flipped:
        y_top, y_bottom = y_start, y_end
    else:
        y_top, y_bottom = y_end, y_start

    p_max  = y_top - y_bottom
    offset = y_bottom

    def model_2p(x_, x0, w):
        z = np.sqrt(2) * (x_ - x0) / w
        return offset + 0.5 * p_max * (1 - special.erf(z) if flipped else 1 + special.erf(z))

    # Initial guess for x0 (midpoint crossing) and w (16%–84% transition half-width)
    interp_ys = ys[::-1] if flipped else ys
    interp_xs = xs[::-1] if flipped else xs

    y_mid = (y_top + y_bottom) / 2.0
    y_16  = y_bottom + 0.16 * p_max
    y_84  = y_bottom + 0.84 * p_max

    x0_guess = float(np.interp(y_mid, interp_ys, interp_xs))
    x_16     = float(np.interp(y_16,  interp_ys, interp_xs))
    x_84     = float(np.interp(y_84,  interp_ys, interp_xs))
    w_guess  = abs(x_84 - x_16) / 2.0

    try:
        popt, _ = curve_fit(model_2p, x, y, p0=[x0_guess, max(w_guess, 1.0)],
                            bounds=([-np.inf, 1e-9], [np.inf, np.inf]))
        x0_fit, w_fit = popt
    except RuntimeError:
        x0_fit, w_fit = x0_guess, w_guess

    r2 = r2_score(y, model_2p(x, x0_fit, w_fit))
    return x0_fit, w_fit, y_top, y_bottom, r2, lambda x_: model_2p(x_, x0_fit, w_fit)


def fit_knife_edge(file_names, data_frames, p0=None, power_scale=1e3, plot=True):
    """
    Fit knife-edge data with error function to extract beam waist at each z position.

    Parameters
    ----------
    file_names   : list of str
    data_frames  : list of DataFrames with 'Time' and 'Power' columns
    p0           : initial guess [x0, p_max, w]; auto-estimated if None
    power_scale  : multiply raw power values by this factor (default 1e3 -> mW)
    plot         : show individual knife-edge fits

    Returns
    -------
    ws : list of fitted beam waists (in time/position units of the data)
    """
    ws = []
    for i, df in enumerate(data_frames):
        x = df['Time'].values
        y = power_scale * df['Power'].values

        if p0 is None:
            p0_i = [np.median(x), np.max(y), (np.max(x) - np.min(x)) / 10]
        else:
            p0_i = p0

        popt, pcov = curve_fit(erf_model, x, y, p0=p0_i)
        yfit = erf_model(x, *popt)

        print(file_names[i])
        print(f'  x0      = {popt[0]:.4f}')
        print(f'  P_max   = {popt[1]:.2f} (scaled units)')
        print(f'  w       = {popt[2]:.4f}')
        print(f'  R²      = {r2_score(y, yfit):.5f}\n')

        ws.append(popt[2])

        if plot:
            fig, ax = plt.subplots()
            ax.plot(x, y, label='Data')
            ax.plot(x, yfit, label='ERF fit')
            ax.set_title(file_names[i])
            ax.set_xlabel('Position / Time')
            ax.set_ylabel('Power (scaled)')
            ax.legend()
            plt.tight_layout()
            plt.show()

    return ws


def waist_vs_z(z, z0, w0, lam=1.064e-6):
    """Gaussian beam waist as a function of propagation distance z."""
    zr = np.pi * w0**2 / lam
    return w0 * np.sqrt(1 + ((z - z0) / zr)**2)


def waist_vs_time(z, z0, w0, a):
    """Waist vs time-axis position (when z is encoded as scan time)."""
    return w0 * np.sqrt(1 + (a * (z - z0) / w0**2)**2)


def fit_waist_profile(z, ws, g0=None, use_time_axis=True, um_conversion=1.0, plot=True):
    """
    Fit the beam waist vs z to extract the minimum waist w0.

    Parameters
    ----------
    z              : array-like, z positions (or scan times)
    ws             : list of waists from fit_knife_edge
    g0             : initial guess [z0, w0, a]; auto-estimated if None
    use_time_axis  : if True, use waist_vs_time; else use waist_vs_z (physical units)
    um_conversion  : multiply fitted waists by this to convert to µm for display
    plot           : show waist profile fit

    Returns
    -------
    popt : optimal fit parameters [z0, w0, (a or lam)]
    """
    model = waist_vs_time if use_time_axis else waist_vs_z

    if g0 is None:
        g0 = [np.mean(z), min(ws), 1.0] if use_time_axis else [np.mean(z), min(ws), 1.064e-6]

    popt, pcov = curve_fit(model, z, ws, g0)
    yfit = model(np.array(z), *popt)

    print('Waist profile fit:')
    print(f'  z0  = {popt[0]:.6f}')
    print(f'  w0  = {popt[1]:.5f}  ->  {popt[1]*um_conversion:.3f} µm')
    if use_time_axis:
        print(f'  a   = {popt[2]:.5f}')
    print(f'  R²  = {r2_score(ws, yfit):.5f}\n')

    if plot:
        z_fine = np.linspace(min(z), max(z), 500)
        ws_fine = [w * um_conversion for w in model(z_fine, *popt)]
        ws_um = [w * um_conversion for w in ws]

        fig, ax = plt.subplots()
        ax.scatter(z, ws_um, color='#FF6103', label='Measured waists', zorder=5)
        ax.plot(z_fine, ws_fine, label='Gaussian fit')
        ax.set_xlabel('z position')
        ax.set_ylabel('Beam waist [µm]')
        ax.set_title('Pulsed Laser Beam Waist Profile')
        ax.legend()
        plt.tight_layout()
        plt.show()

    return popt
