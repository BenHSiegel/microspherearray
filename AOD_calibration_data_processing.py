'''
Processes HDF5 files for the AOD Calibration Data
'''

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit

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
        column_names = ['Power', 'CH0 Freq', 'CH1 Freq', 'CH0 gain','CH0 adjusted gain', 'CH1 gain', 'CH1 adjusted gain', 'Abs time']
        # Convert the data to a pandas dataframe
        df = pd.DataFrame(data, columns=column_names)
        # Remove the first row from the dataframe since it is an artifact of LabView HDF5 file writing
        df = df.iloc[1:].reset_index(drop=True)
    return df

def plot_gain_sweep(df, x_column, y_column, title, x_label, y_label, save_path, save_csv=False):
    '''
    Plots the data
    '''
    # Plot the data
    fig, (ax, ax_residuals) = plt.subplots(2, 1, sharex=True)
    ax.plot(df[x_column], df[y_column], label='Data')
    
    # Add labels and title
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    # Create a secondary y-axis
    ax2 = ax.twinx()
    # Plot the normalized power on the secondary y-axis
    normpower = df[y_column] / (df[y_column].max()/0.90)

    # Perform a linear fit
    slope, intercept = np.polyfit(df[x_column], normpower*100, 1)
    fit_line = slope * df[x_column] + intercept
    ax2.plot(df[x_column], fit_line, label=f'Linear Fit (slope={slope:.4f})', color='green', linestyle='--')

    # Calculate R^2
    residuals = normpower*100 - fit_line
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((normpower*100 - np.mean(normpower*100))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    ax2.plot(df[x_column], normpower*100, label='Normalized Power', color='red')
    ax2.set_ylabel('Normalized Power (%)')
    ax2.legend()
    # Set the number of y-axis tick marks
    ax.yaxis.set_major_locator(plt.MaxNLocator(10))
    ax2.yaxis.set_major_locator(plt.MaxNLocator(10))
    
    ax_residuals.plot(df[x_column], residuals, label='Residuals', color='blue')
    ax_residuals.set_title('Residuals of Linear Fit')
    ax_residuals.set_xlabel(x_label)
    ax_residuals.set_ylabel('Residuals (%)')
    ax_residuals.axhline(0, color='black', linestyle='--')
    ax_residuals.legend()
    
    plt.tight_layout()


    # Save and show the plot
    #fig.savefig(save_path)
    plt.show()
    #plt.close(fig)
    if save_csv:
        # Save df[x_column] and normpower to a CSV file without a header
        output_df = pd.DataFrame({
            x_column: df[x_column]/100,
            'Normalized Power': normpower
        })
        output_df.to_csv(save_path.replace('.png', '.csv'), index=False, header=False)

def plot_freq_map(df, x_column, y_column, z_column, title, x_label, y_label, z_label, save_path):
    '''
    Plots a 2D surface plot of the data point by point
    '''
    # Filter the dataframe to only include points where the z value is greater than 0.01
    df = df[df[z_column] > 0.05]
    # Create a figure and a 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the data point by point
    ax.scatter(df[x_column], df[y_column], df[z_column], c=df[z_column], cmap='viridis')

    # Add labels and title
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label + ' (W)')

    # Add a color bar
    scatter = ax.scatter(df[x_column], df[y_column], df[z_column], c=df[z_column], cmap='viridis')
    fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)

    # Save and show the plot
    #fig.savefig(save_path)

    # Normalize the z_column using its max value
    df[z_column + '_normalized'] = df[z_column] / df[z_column].max()

    # Create a new figure and a 3D axis for the normalized plot
    fig_normalized = plt.figure()
    ax_normalized = fig_normalized.add_subplot(111, projection='3d')

    # Plot the normalized data point by point
    scatter_normalized = ax_normalized.scatter(df[x_column], df[y_column], df[z_column + '_normalized'], c=df[z_column + '_normalized'], cmap='viridis')

    # Add labels and title for the normalized plot
    ax_normalized.set_title(title + ' (Normalized)')
    ax_normalized.set_xlabel(x_label)
    ax_normalized.set_ylabel(y_label)
    ax_normalized.set_zlabel(z_label + ' (Normalized)')

    # Add a color bar for the normalized plot
    fig_normalized.colorbar(scatter_normalized, ax=ax_normalized, shrink=0.5, aspect=5)

    # Save and show the normalized plot
    #fig_normalized.savefig(save_path.replace('.png', '_normalized.png'))
    plt.show()
    # plt.close(fig)
    # plt.close(fig_normalized)


def power_scan_compare(path,filenames):
    counter = 0
    fig, ax = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()
    fig4, ax4 = plt.subplots()
    df_container = [[] for i in range(len(filenames))]
    for filename in filenames:
        file_path = os.path.join(path, filename)

        df = process_hdf5_file(file_path)
        df_container[counter] = df
        difference = (df_container[0]['Power'] - df['Power'])*1000
        ratio = df_container[0]['Power']/df['Power']
        norm_difference = df_container[0]['Power']/df_container[0]['Power'].max() - df['Power']/df['Power'].max()
        norm_ratio = (df_container[0]['Power']/df_container[0]['Power'].max()) / (df['Power']/df['Power'].max())
        ax.plot(df['CH0 gain'],difference,label=filename[10:-3])
        ax2.plot(df['CH0 gain'],ratio,label=filename[10:-3])
        ax3.plot(df['CH0 gain'],norm_difference,label=filename[10:-3])
        ax4.plot(df['CH0 gain'],norm_ratio,label=filename[10:-3])

        counter += 1
    ax.set_xlabel('Channel 0 Gain (%)')
    ax.set_ylabel('Power Difference (mW)')
    ax.legend()
    ax.set_title('Differences in gain action at various frequencies')
    ax2.set_xlabel('Channel 0 Gain (%)')
    ax2.set_ylabel('Power Ratio')
    ax2.legend()
    ax2.set_title('Ratios in gain action at various frequencies')
    ax3.set_xlabel('Channel 0 Gain (%)')
    ax3.set_ylabel('Normalized Power Difference')
    ax3.legend()
    ax3.set_title('Differences in Normalized gain action at various frequencies')
    ax4.set_xlabel('Channel 0 Gain (%)')
    ax4.set_ylabel('Normalized Power Ratio')
    ax4.legend()
    ax4.set_title('Ratios in Normalized gain action at various frequencies')
    plt.show()

    avgpower = df_container[0]['Power']/(df_container[0]['Power'].max()/0.9)
    for i in range(1,len(filenames)):
        avgpower = avgpower + df_container[i]['Power']/(df_container[i]['Power'].max()/0.9)
    avgpower = avgpower/len(filenames)
    # Fit a logarithmic curve to the average power data
    def log_fit(x, a, b):
        return a * np.log(x) + b

    popt, _ = curve_fit(log_fit, df_container[0]['CH0 gain'], avgpower)

    # Plot CH0 gain vs Avg Power
    fig5, ax5 = plt.subplots()
    ax5.plot(df_container[0]['CH0 gain'], avgpower * 100, label='Avg Power')
    # Plot the logarithmic fit
    ax5.plot(df_container[0]['CH0 gain'], log_fit(df_container[0]['CH0 gain'], *popt) * 100, label='Logarithmic Fit', linestyle='--')
    ax5.legend()
    ax5.set_xlabel('Channel 0 Gain (%)')
    ax5.set_ylabel('Gain Observed')
    ax5.set_title('Channel 0 Gain vs Average Power')
    plt.show()
    # Save the CH0 gain and avg power to a CSV file
    # output_df = pd.DataFrame({
    #     'CH0 gain': df_container[0]['CH0 gain']/100,
    #     'Avg Power': avgpower
    # })
    # output_df.to_csv(os.path.join(path, 'avg_power.csv'), index=False)



def main(file_path,folder_path,filename):
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
    plot_gain_sweep(df, 'CH0 adjusted gain', 'Power', filename, 'Channel 0 Gain (%)', 'Power Output (W)', os.path.join(folder_path, 'aod_gain_calibration_data.png'),save_csv=False)
    # plot_freq_map(df, 'CH0 Freq', 'CH1 Freq', 'Power', 'AOD Calibration Data', 'Channel 0 Frequency (Hz)', 'Channel 1 Frequency (Hz)', 'Power Output', 'aod_freq_calibration_data.png')



# path = r'D:\Lab data\20250121'
# filenames = ['gain_scan_20MHz.h5', 'gain_scan_23x20MHz.h5','gain_scan_24-4x22-7MHz.h5',
#              'gain_scan_26-5x20MHz.h5','gain_scan_26-53x27-2MHz.h5','gain_scan_28-1x21-8MHz.h5']
#power_scan_compare(path,filenames)

# path = r'D:\Lab data\20250121\frequency map'
# filename = 'expanded_frequency_map.h5'
# main(os.path.join(path, filename),filename)

path = r'D:\Lab data\20250122'
filename = 'gain_interpolate_check_2.h5'
main(os.path.join(path, filename),path,filename)