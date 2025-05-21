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
import matplotlib as mpl
from matplotlib.cm import ScalarMappable

def process_hdf5_file(file_path, gainorfreq):
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
        column_names = ['Power', 'CH0 Freq', 'CH1 Freq', 'CH0 gain', 'CH0 adjusted gain', 'CH1 gain', 'CH1 adjusted gain', 'Abs time', 'Laser Power']
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
    #df = df[df[z_column] > 0.05]
    # Create a figure and a 3D axis

    fig1, ax1 = plt.subplots()
    ax1.plot(df['Abs time'], df['Laser Power'])
    ax1.set_title('Laser Power vs Time')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Stabilizer PD reading (V but really arb)')


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

    df[z_column + '_pnorm'] = df[z_column] / df['Laser Power']
    
    fig_pnorm = plt.figure()
    ax_pnorm = fig_pnorm.add_subplot(111, projection='3d')
    # Plot the normalized data point by point
    scatter_pnorm = ax_pnorm.scatter(df[x_column], df[y_column], df[z_column + '_pnorm'], c=df[z_column + '_pnorm'], cmap='viridis')

    # Add labels and title for the normalized plot
    ax_pnorm.set_title(title + ' (Power Normalized)')
    ax_pnorm.set_xlabel(x_label)
    ax_pnorm.set_ylabel(y_label)
    ax_pnorm.set_zlabel(z_label + ' (Power Normalized)')

    # Add a color bar for the normalized plot
    fig_pnorm.colorbar(scatter_pnorm, ax=ax_pnorm, shrink=0.5, aspect=5)

    df['comparison'] = df[z_column + '_pnorm']/df[z_column + '_pnorm'].max() - df[z_column + '_normalized']

    fig_comp = plt.figure()
    ax_comp = fig_comp.add_subplot(111, projection='3d')
    # Plot the normalized data point by point
    scatter_comp = ax_comp.scatter(df[x_column], df[y_column], df['comparison'], c=df['comparison'], cmap='viridis')

    # Add labels and title for the normalized plot
    ax_comp.set_title(title + '  (Normalization residuals)')
    ax_comp.set_xlabel(x_label)
    ax_comp.set_ylabel(y_label)
    ax_comp.set_zlabel('Normalized - normalized with fluctuation corrections', labelpad=15)
    ax_comp.tick_params(axis='z', pad=10)

    # Add a color bar for the normalized plot
    fig_comp.colorbar(scatter_comp, ax=ax_comp, shrink=0.5, aspect=5)
    
    # Normalize z_column by the maximum value for each unique (x_column, y_column) pair
    # Here, we normalize z_column by the maximum value for each unique y_column
    df['z_by_ymax'] = df.groupby(y_column)[z_column].transform(lambda x: x / x.max())

    fig_ynorm = plt.figure()
    ax_ynorm = fig_ynorm.add_subplot(111, projection='3d')
    scatter_ynorm = ax_ynorm.scatter(df[x_column], df[y_column], df['z_by_ymax'], c=df['z_by_ymax'], cmap='viridis')

    ax_ynorm.set_title(title + ' (Normalized by max of each ' + y_column + ')')
    ax_ynorm.set_xlabel(x_label)
    ax_ynorm.set_ylabel(y_label)
    ax_ynorm.set_zlabel(z_label + ' (Row-normalized by y)')

    fig_ynorm.colorbar(scatter_ynorm, ax=ax_ynorm, shrink=0.5, aspect=5)

    # Plot the difference of each x value row to the mean of all x value rows
    # Compute the mean z_by_ymax for each x_column across all y_column
    mean_z_by_x = df.groupby(x_column)['z_by_ymax'].mean()
    # Map the mean to each row
    df['mean_z_by_x'] = df[x_column].map(mean_z_by_x)
    # Compute the difference
    df['z_by_ymax_diff_x'] = df['z_by_ymax'] - df['mean_z_by_x']

    fig_diff_x = plt.figure()
    ax_diff_x = fig_diff_x.add_subplot(111, projection='3d')
    scatter_diff_x = ax_diff_x.scatter(df[x_column], df[y_column], df['z_by_ymax_diff_x'], c=df['z_by_ymax_diff_x'], cmap='coolwarm')

    # Recolor points green if z is between -0.01 and 0.01
    cmap = plt.get_cmap('coolwarm')
    normed = (df['z_by_ymax_diff_x'] - df['z_by_ymax_diff_x'].min()) / (df['z_by_ymax_diff_x'].max() - df['z_by_ymax_diff_x'].min())
    colors = np.array([cmap(val) for val in normed])
    mask = (df['z_by_ymax_diff_x'] >= -0.01) & (df['z_by_ymax_diff_x'] <= 0.01)
    colors[mask] = (0, 1, 0, 1)  # RGBA for green

    fig_diff_x = plt.figure()
    ax_diff_x = fig_diff_x.add_subplot(111, projection='3d')
    scatter_diff_x = ax_diff_x.scatter(
        df[x_column], df[y_column], df['z_by_ymax_diff_x'],
        c=colors, marker='o'
    )

    ax_diff_x.set_title(title + ' (Difference to mean of all ' + x_column + ' rows)')
    ax_diff_x.set_xlabel(x_label)
    ax_diff_x.set_ylabel(y_label)
    ax_diff_x.set_zlabel('Difference to mean (Row-normalized by x)')

    # Add a colorbar for only the non-green points
    norm = mpl.colors.Normalize(
        vmin=df['z_by_ymax_diff_x'].min(),
        vmax=df['z_by_ymax_diff_x'].max())
    sm = mpl.cm.ScalarMappable(cmap='coolwarm', norm=norm)
    sm.set_array([])
    fig_diff_x.colorbar(sm, ax=ax_diff_x, shrink=0.5, aspect=5, label='z_by_ymax_diff_x')

    # Now repeat for each unique x_column: normalize z_column by the maximum value for each unique x_column
    df['z_by_xmax'] = df.groupby(x_column)[z_column].transform(lambda x: x / x.max())

    fig_xnorm = plt.figure()
    ax_xnorm = fig_xnorm.add_subplot(111, projection='3d')
    scatter_xnorm = ax_xnorm.scatter(df[x_column], df[y_column], df['z_by_xmax'], c=df['z_by_xmax'], cmap='viridis')

    ax_xnorm.set_title(title + ' (Normalized by max of each ' + x_column + ')')
    ax_xnorm.set_xlabel(x_label)
    ax_xnorm.set_ylabel(y_label)
    ax_xnorm.set_zlabel(z_label + ' (Row-normalized by x)')
    # Filter to only include rows where x and y are between 18 and 32
    mask_xy = (df[x_column] >= 18) & (df[x_column] <= 32) & (df[y_column] >= 18) & (df[y_column] <= 32)
    df_xy = df[mask_xy]

    # Use log scale for z values, avoid log(0) by adding a small epsilon
    epsilon = 1e-8
    z_log = np.log10(df_xy['z_by_xmax'] + epsilon)

    scatter_xnorm = ax_xnorm.scatter(df_xy[x_column], df_xy[y_column], z_log, c=z_log, cmap='viridis')
    ax_xnorm.set_title(title + ' (Log10 Normalized by max of each ' + x_column + ')')
    ax_xnorm.set_xlabel(x_label)
    ax_xnorm.set_ylabel(y_label)
    ax_xnorm.set_zlabel('log10(' + z_label + ' Row-normalized by x)')

    fig_xnorm.colorbar(scatter_xnorm, ax=ax_xnorm, shrink=0.5, aspect=5)

    # Plot the difference of each y value row to the mean of all y value rows, on log scale
    mean_z_by_y = df.groupby(y_column)['z_by_xmax'].mean()
    df['mean_z_by_y'] = df[y_column].map(mean_z_by_y)
    df['z_by_xmax_diff_y'] = df['z_by_xmax'] - df['mean_z_by_y']

    df_xy = df[mask_xy]

    # Use log scale for the difference, add epsilon to avoid log(0)
    z_diff_log = np.log10(np.abs(df_xy['z_by_xmax_diff_y']) + epsilon) * np.sign(df_xy['z_by_xmax_diff_y'])

    fig_diff_y = plt.figure()
    ax_diff_y = fig_diff_y.add_subplot(111, projection='3d')
    scatter_diff_y = ax_diff_y.scatter(df_xy[x_column], df_xy[y_column], z_diff_log, c=z_diff_log, cmap='coolwarm')

    # Recolor points green if z_diff_log is between -0.01 and 0.01
    cmap_y = plt.get_cmap('coolwarm')
    normed_y = (z_diff_log - z_diff_log.min()) / (z_diff_log.max() - z_diff_log.min())
    colors_y = np.array([cmap_y(val) for val in normed_y])
    mask_y = (z_diff_log >= -0.01) & (z_diff_log <= 0.01)
    colors_y[mask_y] = (0, 1, 0, 1)  # RGBA for green

    fig_diff_y = plt.figure()
    ax_diff_y = fig_diff_y.add_subplot(111, projection='3d')
    scatter_diff_y = ax_diff_y.scatter(
        df_xy[x_column], df_xy[y_column], z_diff_log,
        c=colors_y, marker='o'
    )

    ax_diff_y.set_title(title + ' (Log10 Difference to mean of all ' + y_column + ' rows)')
    ax_diff_y.set_xlabel(x_label)
    ax_diff_y.set_ylabel(y_label)
    ax_diff_y.set_zlabel('log10(Difference to mean) (Row-normalized by y)')

    norm_y = mpl.colors.Normalize(
        vmin=z_diff_log.min(),
        vmax=z_diff_log.max())
    sm_y = mpl.cm.ScalarMappable(cmap='coolwarm', norm=norm_y)
    sm_y.set_array([])
    fig_diff_y.colorbar(sm_y, ax=ax_diff_y, shrink=0.5, aspect=5, label='log10(z_by_xmax_diff_y)')

    plt.show()
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
    df = process_hdf5_file(file_path,'freq')
    # Plot the data
    #plot_gain_sweep(df, 'CH0 adjusted gain', 'Power', filename, 'Channel 0 Gain (%)', 'Power Output (W)', os.path.join(folder_path, 'aod_gain_calibration_data.png'),save_csv=False)
    plot_freq_map(df, 'CH0 Freq', 'CH1 Freq', 'Power', 'AOD Calibration Data', 'Channel 0 Frequency (Hz)', 'Channel 1 Frequency (Hz)', 'Power Output', 'aod_freq_calibration_data.png')



# path = r'D:\Lab data\20250121'
# filenames = ['gain_scan_20MHz.h5', 'gain_scan_23x20MHz.h5','gain_scan_24-4x22-7MHz.h5',
#              'gain_scan_26-5x20MHz.h5','gain_scan_26-53x27-2MHz.h5','gain_scan_28-1x21-8MHz.h5']
#power_scan_compare(path,filenames)

# path = r'D:\Lab data\20250121\frequency map'
# filename = 'expanded_frequency_map.h5'
# main(os.path.join(path, filename),filename)

path = r'D:\Lab data\20250520'
filename = 'AOD frequency sweep.h5'
main(os.path.join(path, filename),path,filename)