import h5py
import numpy as np

def hdf5_sphere_data_scraper(filename):
    '''
    Basic hdf5 reader for the sphere data files.
    filename = path to the file you want to read
    Outputs:
    xposdata = numpy array where each column in the array is the x timestream data from a sphere
    yposdata = numpy array where each column in the array is the y timestream data from a sphere
    xfftmatrix = numpy array where each column in the array is the x amplitude spectral density data from a sphere (in m/root(Hz))
    yfftmatrix = numpy array where each column in the array is the y amplitude spectral density data from a sphere (in m/root(Hz))
    frequency_bins = 1D array containing the frequency bins that were used in the PSD calculations
    '''
    #Opens the HDF5 file in read mode
    hf = h5py.File(filename, 'r')
    
    #In hdf5 files, it is like a file system where groups are folders and datasets are the files inside them
    #Attributes are little info tags that you can link onto group or datasets
    #Reads the "position" group of file
    posgroup = hf.get('position')
    #Reads the X_psd group which has the frequency bins and amplitude spectral density values for each sphere saved
    xpsdgroup = hf.get('X_psd')
    #Reads the Y_psd group which has the frequency bins and amplitude spectral density values for each sphere saved
    ypsdgroup = hf.get('Y_psd')
    #reads the framerate attribute from the group
    fs = posgroup.attrs['framerate (fps)']

    #create some empty lists to read the data into
    xposdata = []
    yposdata = []
    xfftmatrix = []
    yfftmatrix = []
    frequency_bins = []

    #make a loop counter to initialize the lists properly on first call
    l=0
    #Uses group.items() to see all the things inside of this group and iterate through them
    #For the sphere video hdf5 file, the items in this group are the position data sets for each sphere
    for j in posgroup.items():
        #read the data from each item the np.array(item[1]) will take the data and convert it into a numpy array
        pos = np.array(j[1])
        #Takes the second column (which is the x position data) and reads it into a temporary variable
        xpos = pos[:,1].reshape(-1,1)
        #Takes the third column (which is the y position data) and reads it into a temporary variable
        ypos = pos[:,2].reshape(-1,1)
        
        #initializes the position data array
        if l == 0:
            xposdata = xpos[:,0].reshape(-1,1)
            yposdata = ypos[:,0].reshape(-1,1)
        #concatenates the next sphere's position data to the array
        else:
            xposdata = np.concatenate((xposdata, xpos[:,0].reshape(-1,1)), axis=1)
            yposdata = np.concatenate((yposdata, ypos[:,0].reshape(-1,1)), axis=1)
        
        l+=1
            
    #makes a counter variable to keep track of what item of the group we are on    
    k=0
    for j in xpsdgroup.items():
        #reads the items data into a numpy array
        xfftj = np.array(j[1])
        #initializes the inner list in the list of lists
        if k == 0:
            xfftmatrix = xfftj[:,1].reshape(-1,1) #reshapes the data so that it can be concatenated to later
        else:
            xfftmatrix = np.concatenate((xfftmatrix, xfftj[:,1].reshape(-1,1)), axis=1)
        k+=1

    #see above, but runs through the Y psd data now   
    k=0
    for j in ypsdgroup.items():
        yfftj = np.array(j[1])
        if k == 0:
            yfftmatrix = yfftj[:,1].reshape(-1,1)
        else:
            yfftmatrix = np.concatenate((yfftmatrix, yfftj[:,1].reshape(-1,1)), axis=1)
        
        #rewrites over frequency_bins (it doesn't matter whether we used the x or y frequency, or from which sphere. They should all be the same)
        #of course not a great way to do it since I just keep rewriting over it
        frequency_bins = yfftj[:,0]
        k+=1

    #close the hdf5 file once complete   
    hf.close()

    return xposdata, yposdata, xfftmatrix, yfftmatrix, frequency_bins, fs