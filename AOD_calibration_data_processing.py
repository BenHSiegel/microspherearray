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

    #Plot the laser power vs time
    fig1, ax1 = plt.subplots()
    ax1.plot(df['Abs time'], df['Laser Power'])
    ax1.set_title('Laser Power vs Time')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Stabilizer PD reading (V but really arb)')

    # Plot the normalized power and fits
    fig2, ax2 = plt.subplots()
    ax2.set_title(title)
    ax2.set_xlabel(x_label)
    ax2.set_ylabel('Normalized Power (%)')
    df['normpower'] = df[y_column] / (df[y_column].max())
    df['lasernorm'] = df[y_column] / df['Laser Power']
    df['lasernorm'] = df['lasernorm'] / (df['lasernorm'].max())
    df['lasernorm2'] = df['normpower'] * (df['Laser Power'].max()) / df['Laser Power']
    ax2.plot(df[x_column], df['lasernorm']*100, label='Laser Power Normalized', color='orange')
    ax2.plot(df[x_column], df['lasernorm2']*100, label='Laser Power Normalized 2', color='purple')
    # Perform a linear fit
    n = len(df)
    start = int(n * 0.2)
    end = int(n * 0.8)
    fit_x = df[x_column].iloc[start:end]
    fit_y = (df['lasernorm'] * 100).iloc[start:end]
    slope, intercept = np.polyfit(fit_x, fit_y, 1)
    fit_line = slope * df[x_column] + intercept
    ax2.plot(df[x_column], fit_line, label=f'Linear Fit (slope={slope:.4f})', color='green', linestyle='--')
    # Add dotted lines at y = 0 and y = 100
    ax2.axhline(0, color='gray', linestyle=':', linewidth=1)
    ax2.axhline(100, color='gray', linestyle=':', linewidth=1)
    ax2.plot(df[x_column], df['normpower']*100, label='Normalized Power', color='red')
    ax2.yaxis.set_major_locator(plt.MaxNLocator(10))
    ax2.legend()

    # Residuals figure
    fig_residuals, ax_residuals = plt.subplots()
    residuals = df['normpower']*100 - fit_line
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((df['normpower']*100 - np.mean(df['normpower']*100))**2)
    r_squared = 1 - (ss_res / ss_tot)
    ax_residuals.plot(df[x_column], residuals, label='Residuals', color='blue')
    ax_residuals.set_title('Residuals')
    ax_residuals.set_xlabel(x_label)
    ax_residuals.set_ylabel('Residuals (%)')
    ax_residuals.legend()

    # Errors figure
    fig_errors, ax_errors = plt.subplots()
    residuals_laser = df['lasernorm']*100 - df['normpower']*100
    residuals_laser2 = df['lasernorm2']*100 - df['normpower']*100
    residuals_laser3 = df['lasernorm2']*100 - df['lasernorm']*100
    ax_errors.plot(df[x_column], residuals_laser, label='Error calc 1', color='blue')
    ax_errors.plot(df[x_column], residuals_laser2, label='Error calc 2', color='orange')
    ax_errors.plot(df[x_column], residuals_laser3, label='Error 2-1', color='purple')
    ax_errors.set_title('Errors based on laser fluctuations')
    ax_errors.set_xlabel(x_label)
    ax_errors.set_ylabel('Residuals (%)')
    ax_errors.axhline(0, color='black', linestyle='--')
    ax_errors.legend()

    plt.tight_layout()
    plt.show()
    #plt.close(fig)
    if save_csv:
        # Save df[x_column] and normpower to a CSV file without a header
        output_df = pd.DataFrame({
            x_column: df[x_column]/100,
            'Normalized Power': df['lasernorm']
        })
        output_df.to_csv(save_path.replace('.png', '.csv'), index=False, header=False)

def plot_freq_map(df, x_column, y_column, z_column, title, x_label, y_label, z_label, save_path, save_csv=False):
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

    ###########################################################################################
    # Create a new figure and a 3D axis for the normalized plot

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

    ####################################################################################################################
    #Adjusts the measured laser power to account for the laser power fluctuations measured by the stabilizer

    df[z_column + '_padjust'] = df[z_column] / df['Laser Power']
    
    fig_pnorm = plt.figure()
    ax_pnorm = fig_pnorm.add_subplot(111, projection='3d')
    # Plot the normalized data point by point
    scatter_pnorm = ax_pnorm.scatter(df[x_column], df[y_column], df[z_column + '_padjust'], c=df[z_column + '_padjust'], cmap='viridis')

    # Add labels and title for the normalized plot
    ax_pnorm.set_title(title + ' (Adjusted)')
    ax_pnorm.set_xlabel(x_label)
    ax_pnorm.set_ylabel(y_label)
    ax_pnorm.set_zlabel(z_label + ' (Adjusted)')

    # Add a color bar for the normalized plot
    fig_pnorm.colorbar(scatter_pnorm, ax=ax_pnorm, shrink=0.5, aspect=5)

    ###########################################################################################################
    # Calculate the difference between the normalized z_column and the normalized z_column with fluctuation corrections

    df[z_column + '_adjustnorm'] = df[z_column + '_padjust']/df[z_column + '_padjust'].max()
    df['comparison'] = df[z_column + '_adjustnorm'] - df[z_column + '_normalized']

    fig_comp = plt.figure()
    ax_comp = fig_comp.add_subplot(111, projection='3d')
    # Plot the normalized data point by point
    scatter_comp = ax_comp.scatter(df[x_column], df[y_column], df['comparison'], c=df['comparison'], cmap='viridis')

    # Add labels and title for the normalized plot
    ax_comp.set_title(title + '  (Normalization Comparison)')
    ax_comp.set_xlabel(x_label)
    ax_comp.set_ylabel(y_label)
    ax_comp.set_zlabel('Normalized - Adjusted and Normalized', labelpad=15)
    ax_comp.tick_params(axis='z', pad=10)

    # Add a color bar for the normalized plot
    fig_comp.colorbar(scatter_comp, ax=ax_comp, shrink=0.5, aspect=5)
    
    ##########################################################################################################

    # Normalize the adjusted power values by the maximum value for each unique (x_column, y_column) pair
    # Here, we normalize adjusted power by the maximum value for each unique y_column
    df['z_by_ymax'] = df.groupby(y_column)[z_column + '_padjust'].transform(lambda x: x / x.max())

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
    median_z_by_x = df.groupby(x_column)['z_by_ymax'].median()
    print(median_z_by_x)
    # Map the median to each row
    df['median_z_by_x'] = df[x_column].map(median_z_by_x)
    # Compute the difference
    df['z_by_ymax_diff_x'] = df['z_by_ymax'] - df['median_z_by_x']

    # fig_diff_x = plt.figure()
    # ax_diff_x = fig_diff_x.add_subplot(111, projection='3d')
    # scatter_diff_x = ax_diff_x.scatter(df[x_column], df[y_column], df['z_by_ymax_diff_x'], c=df['z_by_ymax_diff_x'], cmap='coolwarm')

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
    ax_diff_x.set_zlabel('Difference to mean (Row-normalized by y)')

    # Add a colorbar for only the non-green points
    norm = mpl.colors.Normalize(
        vmin=df['z_by_ymax_diff_x'].min(),
        vmax=df['z_by_ymax_diff_x'].max())
    sm = mpl.cm.ScalarMappable(cmap='coolwarm', norm=norm)
    sm.set_array([])
    fig_diff_x.colorbar(sm, ax=ax_diff_x, shrink=0.5, aspect=5, label='z_by_ymax_diff_x')

    # Now repeat for each unique x_column: normalize z_column by the maximum value for each unique x_column
    # Normalize the adjusted power values by the maximum value for each unique x_column
    df['z_by_xmax'] = df.groupby(x_column)[z_column + '_padjust'].transform(lambda x: x / x.max())

    fig_xnorm = plt.figure()
    ax_xnorm = fig_xnorm.add_subplot(111, projection='3d')
    scatter_xnorm = ax_xnorm.scatter(df[x_column], df[y_column], df['z_by_xmax'], c=df['z_by_xmax'], cmap='viridis')

    ax_xnorm.set_title(title + ' (Normalized by max of each ' + x_column + ')')
    ax_xnorm.set_xlabel(x_label)
    ax_xnorm.set_ylabel(y_label)
    ax_xnorm.set_zlabel(z_label + ' (Row-normalized by x)')

    fig_xnorm.colorbar(scatter_xnorm, ax=ax_xnorm, shrink=0.5, aspect=5)

    # Plot the difference of each y value row to the mean of all y value rows
    median_z_by_y = df.groupby(y_column)['z_by_xmax'].median()
    df['median_z_by_y'] = df[y_column].map(median_z_by_y)
    df['z_by_xmax_diff_y'] = df['z_by_xmax'] - df['median_z_by_y']

    cmap2 = plt.get_cmap('coolwarm')
    normed2 = (df['z_by_xmax_diff_y'] - df['z_by_xmax_diff_y'].min()) / (df['z_by_xmax_diff_y'].max() - df['z_by_xmax_diff_y'].min())
    colors2 = np.array([cmap2(val) for val in normed2])
    mask2 = (df['z_by_xmax_diff_y'] >= -0.01) & (df['z_by_xmax_diff_y'] <= 0.01)
    colors2[mask2] = (0, 1, 0, 1)  # RGBA for green

    fig_diff_y = plt.figure()
    ax_diff_y = fig_diff_y.add_subplot(111, projection='3d')
    scatter_diff_y = ax_diff_y.scatter(
        df[x_column], df[y_column], df['z_by_xmax_diff_y'],
        c=colors2, marker='o'
    )

    ax_diff_y.set_title(title + ' (Difference to mean of all ' + y_column + ' rows)')
    ax_diff_y.set_xlabel(x_label)
    ax_diff_y.set_ylabel(y_label)
    ax_diff_y.set_zlabel('Difference to mean (Row-normalized by x)')

    norm2 = mpl.colors.Normalize(
        vmin=df['z_by_xmax_diff_y'].min(),
        vmax=df['z_by_xmax_diff_y'].max())
    sm2 = mpl.cm.ScalarMappable(cmap='coolwarm', norm=norm2)
    sm2.set_array([])
    fig_diff_y.colorbar(sm2, ax=ax_diff_y, shrink=0.5, aspect=5, label='z_by_xmax_diff_y')

    if save_csv:
        # Save a CSV file with the x values and median z by x
        output_df = pd.DataFrame({
            x_column: median_z_by_x.index,
            'median_z_by_x': median_z_by_x.values
        })
        output_df.to_csv(save_path.replace('.png', '_CH0calibration.csv'), index=False, header=False)

        # Save a CSV file with the y values and median z by y
        output_df_y = pd.DataFrame({
            y_column: median_z_by_y.index,
            'median_z_by_y': median_z_by_y.values
        })
        output_df_y.to_csv(save_path.replace('.png', '_CH1calibration.csv'), index=False, header=False)
    
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



def main(file_path, folder_path, filename, plot_type='freq_map',save_csv=False):
    '''
    Main function
    plot_type: 'gain_sweep' or 'freq_map'
    '''
    # Check if the file exists
    if not os.path.exists(file_path):
        print('File does not exist')
        sys.exit(1)
    # Process the HDF5 file
    df = process_hdf5_file(file_path, 'freq')
    # Choose which plot to run
    if plot_type == 'gain_sweep':
        plot_gain_sweep(
            df, 'CH1 gain', 'Power', filename,
            'Channel 1 Gain (%)', 'Power Output (W)',
            os.path.join(folder_path, 'aod_gain_calibration_data.png'),
            save_csv=False
        )
    elif plot_type == 'freq_map':
        plot_freq_map(
            df, 'CH0 Freq', 'CH1 Freq', 'Power',
            'AOD Calibration Data',
            'Channel 0 Frequency (Hz)', 'Channel 1 Frequency (Hz)', 'Power Output',
            os.path.join(folder_path, 'aod_gain_calibration_data.png'),
            save_csv=True
        )
    else:
        print(f"Unknown plot_type: {plot_type}")
        sys.exit(1)



# path = r'D:\Lab data\20250121'
# filenames = ['gain_scan_20MHz.h5', 'gain_scan_23x20MHz.h5','gain_scan_24-4x22-7MHz.h5',
#              'gain_scan_26-5x20MHz.h5','gain_scan_26-53x27-2MHz.h5','gain_scan_28-1x21-8MHz.h5']
#power_scan_compare(path,filenames)


path = r'D:\Lab data\20250522'
filename = 'test_CH1freq_fix.h5'
main(os.path.join(path, filename),path,filename, plot_type='freq_map')