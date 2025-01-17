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
        # Get the column names
        column_names = list(data.dtype.names)
        # Convert the data to a pandas dataframe
        df = pd.DataFrame(data, columns=column_names)
    return df

def plot_data(df, x_column, y_column, title, x_label, y_label, save_path):
    '''
    Plots the data
    '''
    fig, ax = plt.subplots()
    ax.plot(df[x_column], df[y_column], 'o')
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    fig.savefig(save_path)
    plt.close(fig)

def main():
    '''
    Main function
    '''
    # Check the number of arguments
    if len(sys.argv) != 2:
        print('Usage: python3 process_hdf5.py <file_path>')
        sys.exit(1)
    # Get the file path
    file_path = sys.argv[1]
    # Check if the file exists
    if not os.path.exists(file_path):
        print('File does not exist')
        sys.exit(1)
    # Process the HDF5 file
    df = process_hdf5_file(file_path)
    # Plot the data
    plot_data(df, 'wavelength', 'aod', 'AOD Calibration Data', 'Wavelength (nm)', 'AOD', 'aod_calibration_data.png')

if __name__ == '__main__':
    main()

