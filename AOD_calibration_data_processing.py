'''
Processes HDF5 files for the AOD Calibration Data
'''

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

def process_hdf5_file(file_path):
    '''
    Processes an HDF5 file and returns a pandas dataframe with the data
    '''
    # Open the HDF5 file
    with h5py.File(file_path, 'r') as file:
        # Get the keys
        keys = list(file.keys())
        # Get the data
        data = file[keys[0]][()]
        # Get the number of columns
        num_columns = data.shape[1]
        # Generate column names based on column numbers
        column_names = ['Power', 'CH0 Freq', 'CH1 Freq', 'CH0 gain', 'CH1 gain', 'Abs time']
        # Convert the data to a pandas dataframe
        df = pd.DataFrame(data, columns=column_names)
        # Remove the first row from the dataframe since it is an artifact of LabView HDF5 file writing
        df = df.iloc[1:].reset_index(drop=True)
    return df

def plot_data(df, x_column, y_column, title, x_label, y_label, save_path):
    '''
    Plots the data
    '''
    # Fit a linear model to the first third of the data
    first_third = df.iloc[:len(df) // 3]
    coefficients = np.polyfit(first_third[x_column], first_third[y_column], 1)
    linear_fit = np.poly1d(coefficients)
    slope = coefficients[0]
    intercept = coefficients[1]
    residuals = first_third[y_column] - linear_fit(first_third[x_column])
    residual_sum_of_squares = np.sum(residuals ** 2)

    # Plot the data
    fig, ax = plt.subplots()
    ax.plot(df[x_column], df[y_column], label='Data')
    ax.plot(first_third[x_column], linear_fit(first_third[x_column]), label=f'Linear Fit (slope={slope:.2f}, Residual={residual_sum_of_squares:.5f})', linestyle='--')

    # Add labels and title
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()

    # Save and show the plot
    #fig.savefig(save_path)
    plt.show()
    #plt.close(fig)


def main(file_path):
    '''
    Main function
    '''
    # Check if the file exists
    if not os.path.exists(file_path):
        print('File does not exist')
        sys.exit(1)
    # Process the HDF5 file
    df = process_hdf5_file(file_path)
    # Plot the data
    plot_data(df, 'CH0 gain', 'Power', 'AOD Calibration Data', 'Channel 0 Gain (%)', 'Power Output (W)', 'aod_calibration_data.png')


filename = r'D:\Lab data\20250117\ch0_power_scan_2.h5'
main(filename)
