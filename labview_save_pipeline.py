import os
from collections import defaultdict
import re
import pandas as pd
import numpy as np

class FileGrouper:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.groups = defaultdict(list)
        self.pattern = re.compile(r'^(\d+Hz-\d+gain-\d+)')

    def group_files(self):
        if not os.path.isdir(self.folder_path):
            raise ValueError(f"{self.folder_path} is not a valid directory.")

        for file_name in os.listdir(self.folder_path):
            match = self.pattern.match(file_name)
            if match:
                key = match.group(1)
                self.groups[key].append(file_name)

    def process_groups(self):
        results = {}
        for group_key, files in self.groups.items():
            psd_list = []
            for file_name in files:
                file_path = os.path.join(self.folder_path, file_name)
                df = self.read_file(file_path)  # Placeholder function to read the file
                psd = self.compute_psd(df)  # Compute PSD for the dataframe
                psd_list.append(psd)
            # Average PSDs across all files in the group
            avg_psd = np.mean(psd_list, axis=0)
            # Compute the ratio of each row's PSD to the PSDs of other rows
            ratio_matrix = self.compute_ratios(avg_psd)
            # Extract drive_frequency from the group key
            drive_frequency = int(group_key.split('Hz')[0])

            # Compute the phase difference between each row's data and the data in other rows
            phase_diff_matrix = self.compute_phase_differences(psd_list)

            # Process the average PSDs based on the two options
            riemann_sum = self.compute_riemann_sum(avg_psd, drive_frequency)
            closest_bin_value = self.get_closest_bin_value(avg_psd, drive_frequency)

            # Process the ratios based on the two options
            ratio_riemann_sum = self.compute_riemann_sum(ratio_matrix, drive_frequency)
            ratio_closest_bin_value = self.get_closest_bin_value(ratio_matrix, drive_frequency)

            results[group_key] = {
                "ratios": {
                    "riemann_sum": ratio_riemann_sum,
                    "closest_bin_value": ratio_closest_bin_value
                },
                "phase_differences": phase_diff_matrix,
                "average_psd": {
                    "riemann_sum": riemann_sum,
                    "closest_bin_value": closest_bin_value
                }
            }
        return results

    def compute_riemann_sum(self, data, drive_frequency, range_width=10):
        # Compute the Riemann sum for the data in a set range around drive_frequency
        freq_bins = np.arange(len(data))  # Assuming frequency bins are indexed
        lower_bound = max(0, drive_frequency - range_width)
        upper_bound = min(len(data), drive_frequency + range_width)
        indices_in_range = (freq_bins >= lower_bound) & (freq_bins <= upper_bound)
        return np.sum(data[indices_in_range])

    def get_closest_bin_value(self, data, drive_frequency):
        # Get the value in the frequency bin closest to drive_frequency
        freq_bins = np.arange(len(data))  # Assuming frequency bins are indexed
        closest_bin = np.argmin(np.abs(freq_bins - drive_frequency))
        return data[closest_bin]

import matplotlib.pyplot as plt

def plot_results(self, results):
    # Extract data for plotting
    drive_frequencies = []
    ratio_riemann_sums = []
    phase_differences = []

    for group_key, data in results.items():
        drive_frequency = int(group_key.split('Hz')[0])
        drive_frequencies.append(drive_frequency)
        ratio_riemann_sums.append(data["ratios"]["riemann_sum"])
        phase_differences.append(np.mean(data["phase_differences"]))  # Average phase differences

    # Plot ratio Riemann sums
    plt.figure(figsize=(10, 5))
    plt.plot(drive_frequencies, ratio_riemann_sums, marker='o', label="Ratio Riemann Sum")
    plt.xlabel("Drive Frequency (Hz)")
    plt.ylabel("Ratio Riemann Sum")
    plt.title("Ratio Riemann Sum vs Drive Frequency")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Plot phase differences
    plt.figure(figsize=(10, 5))
    plt.plot(drive_frequencies, phase_differences, marker='o', label="Phase Differences")
    plt.xlabel("Drive Frequency (Hz)")
    plt.ylabel("Average Phase Difference")
    plt.title("Phase Differences vs Drive Frequency")
    plt.grid(True)
    plt.legend()
    plt.show()

    return