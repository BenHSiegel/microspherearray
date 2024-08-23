# Libraries
import pandas as pd
import numpy as np
import scipy as sp
import math 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import special
import os
from sklearn.metrics import r2_score

# Reading multiple csv files

path = r"D:\Lab data\20240821\Knive Edge Inside Chamber"
os.chdir(path)
print(os.getcwd())
file_name_directory=[]
file_directory=[]

for filename in sorted(os.listdir(path)):
    if filename.endswith(".txt"):
        print(filename)
        file_name_directory.append(filename)
        df = pd.read_csv(filename, delimiter='\t', header=32,  usecols=[0, 1], names=['Time', 'Power'])
        file_directory.append(df)

#del file_directory[1]


# define function for curve fitting - - error function
def model(x,x0,p_max,w):
    return (1/2)*p_max*(1+special.erf(np.sqrt(2)*(x-x0)/w))

#initial guess for parameters: x0, p_max, w
pO=[50,750,50] 

ws=[]
  
i=0      
for df in file_directory:
    #length = len(df['Time'])
    #x = np.arange(length)
    x = df['Time']
    
    y = 10**3*df['Power']
    
    # fit curve 
    popt, pcov = curve_fit(model,x,y,pO) #popt -- will give optimal values of parameters
    
    #define the fitting function
    yp=model(x,popt[0],popt[1],popt[2])
    print(file_name_directory[i])
    print('x0 = %.2f s'% (popt[0]))
    print('Maximum power = {:.2f} mW'.format(popt[1]))
    print('Beam waist = {:.2f} s'.format(popt[2]))
    print('R^2 : %.5f'%(r2_score(y,yp)))
    print('\n')
    
    w=popt[2]
    ws.append(w)
    
    # plt.figure()
    # plt.title('Knife edge')
    # plt.xlabel('Time')
    # plt.ylabel('Optical power (mW)')
    # plt.plot(x,y,label='Data')
    # plt.plot(x,yp, label='Fit')
    
    # plt.legend()
    
    i=i+1
    
    
    #Define waist
def waist_dist(z, z0, w0):
    l = 1.064e-6 #wavelength of beam
    zr = sp.pi*w0**2/l #raleigh length
    return w0*np.sqrt(1 + ((z-z0)/zr)**2) 

def waist_time(z,z0,w0,a):
    return w0*np.sqrt(1 + (a*(z-z0)/w0**2)**2) 

#NEED TO CHANGE VALUES DEPENDING ON HOW MANY Z's TAKEN
z = [3,4,5,6,7,8,9,10]


#initial guess for z0,w0,a
g1=[6.5,5, -150] 

popt, pcov = curve_fit(waist_time,z,ws,g1)

#define the fitting function
yp_w=waist_time(z,popt[0], popt[1],popt[2])

print("The beam's waist profile is:")
print('z0 = %.8f s'% (popt[0]))
print('w0 = %.5f s'% (popt[1]))
print('a = %.5f s'% (popt[2]))

# find R^2 value
print('R^2 : %.5f'%(r2_score(ws,yp_w)))
plt.figure()
#plt.title('Gaussian fit using Knife Edge method')
plt.xlabel('z=Time')
plt.ylabel('w')

#Plot data
plt.plot(z,ws, label='from erf', marker=".")

#Plot the fitting function
plt.plot(z,yp_w, label='fit', marker=".")
plt.legend()

z0_fit = popt[0]
w0_fit = popt[1]
a_fit = popt[2]

line_guess = []

z_guess = np.linspace(min(z),max(z),1000)

for z_val in z_guess:
    line = waist_time(z_val,z0_fit,w0_fit,a_fit) *1.75 #includes the factor to convert seconds in slow mode to um
    line_guess.append(line)


wum = [x * 1.769 for x in ws]

print('w0 = %.5f um'% (popt[1]*1.75))
plt.figure()
plt.plot(z_guess,line_guess, label='Fit')  
plt.scatter(z,wum, c= '#FF6103', label='From ERFs')
plt.legend()
plt.ylabel("Beam Waist [um]")
plt.xlabel("Distance from the bottom of the stage's motion [s]")
plt.title('Waist position inside chamber')
plt.show()
    