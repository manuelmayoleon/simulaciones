# trans_eq
import numpy as np
import pandas as pd
import scipy.special as special
from numba import jit
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True
from matplotlib.transforms import (
    Bbox, TransformedBbox, blended_transform_factory)
import random
# import scipy.special as special

# Import math Library
import math 
import numpy.ma as ma
from sympy import *

import csv
import os 
import glob


from scipy.optimize import curve_fit

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


path = os.getcwd()
path = path+"/Datos/"
csv_files = glob.glob(os.path.join(path, "*.dat"))
csv_files = sorted(csv_files, key=len)
# print(len(csv_files))

names = list()
elements=list()
# loop over the list of csv files
for ii in csv_files:
        
        # read the csv file
        # df= pd.read_csv(f)
       
        df= pd.read_csv(ii,header=None,sep='\s+',engine='python')
        elements.append(df)
        # print the location, filename and name
        # print('Location:', ii)
        file_name = ii.split("/")[-1]
        # print('File Name:', file_name )
        name = file_name.split(".dat")[0]
        name=file_name
        print('Name:', name)
        names.append(name)
          
        # print the content
        # print('Content:')
        # print(df)  
# print(elements) 
# print(names)

# !! rho para tau = 1.0

r015 =  elements[names.index('NormDensidadEscalon-1.0-0.1-0.5.dat')]  
r014 =  elements[names.index('NormDensidadEscalon-1.0-0.1-0.4.dat')] 
r013 =  elements[names.index('NormDensidadEscalon-1.0-0.1-0.3.dat')]  
r012 =  elements[names.index('NormDensidadEscalon-1.0-0.1-0.2.dat')]  
r011 =  elements[names.index('NormDensidadEscalon-1.0-0.1-0.1.dat')]  
r0110 =  elements[names.index('NormDensidadEscalon-1.0-0.1-1.0.dat')]  




r02 =  elements[names.index('NormDensidadEscalon-1.0-0.2-0.4.dat')]   
r021 =  elements[names.index('NormDensidadEscalon-1.0-0.2-0.1.dat')]   
r022 =  elements[names.index('NormDensidadEscalon-1.0-0.2-0.2.dat')]   
r023 =  elements[names.index('NormDensidadEscalon-1.0-0.2-0.3.dat')]   
r025 =  elements[names.index('NormDensidadEscalon-1.0-0.2-0.5.dat')]   
r0210 =  elements[names.index('NormDensidadEscalon-1.0-0.2-1.0.dat')]  



r045 =  elements[names.index('NormDensidadEscalon-1.0-0.4-0.5.dat')]  
r044 =  elements[names.index('NormDensidadEscalon-1.0-0.4-0.4.dat')] 
r043 =  elements[names.index('NormDensidadEscalon-1.0-0.4-0.3.dat')]  
r042 =  elements[names.index('NormDensidadEscalon-1.0-0.4-0.2.dat')]  
r041 =  elements[names.index('NormDensidadEscalon-1.0-0.4-0.1.dat')]  
r0410 =  elements[names.index('NormDensidadEscalon-1.0-0.4-1.0.dat')]


r055 =  elements[names.index('NormDensidadEscalon-1.0-0.5-0.5.dat')]  
r054 =  elements[names.index('NormDensidadEscalon-1.0-0.5-0.4.dat')] 
r053 =  elements[names.index('NormDensidadEscalon-1.0-0.5-0.3.dat')]  
r052 =  elements[names.index('NormDensidadEscalon-1.0-0.5-0.2.dat')]  
r051 =  elements[names.index('NormDensidadEscalon-1.0-0.5-0.1.dat')]  
r0510 =  elements[names.index('NormDensidadEscalon-1.0-0.5-1.0.dat')]


r075 =  elements[names.index('NormDensidadEscalon-1.0-0.7-0.5.dat')]  
r074 =  elements[names.index('NormDensidadEscalon-1.0-0.7-0.4.dat')] 
r073 =  elements[names.index('NormDensidadEscalon-1.0-0.7-0.3.dat')]  
r072 =  elements[names.index('NormDensidadEscalon-1.0-0.7-0.2.dat')]  
r071 =  elements[names.index('NormDensidadEscalon-1.0-0.7-0.1.dat')]  
r0710 =  elements[names.index('NormDensidadEscalon-1.0-0.7-1.0.dat')]



r075 =  elements[names.index('NormDensidadEscalon-1.0-0.7-0.5.dat')]  
r074 =  elements[names.index('NormDensidadEscalon-1.0-0.7-0.4.dat')] 
r073 =  elements[names.index('NormDensidadEscalon-1.0-0.7-0.3.dat')]  
r072 =  elements[names.index('NormDensidadEscalon-1.0-0.7-0.2.dat')]  
r071 =  elements[names.index('NormDensidadEscalon-1.0-0.7-0.1.dat')]  
r0710 =  elements[names.index('NormDensidadEscalon-1.0-0.7-1.0.dat')]

r0105 =  elements[names.index('NormDensidadEscalon-1.0-1.0-0.5.dat')]  
r0104 =  elements[names.index('NormDensidadEscalon-1.0-1.0-0.4.dat')] 
r0103 =  elements[names.index('NormDensidadEscalon-1.0-1.0-0.3.dat')]  
r0102 =  elements[names.index('NormDensidadEscalon-1.0-1.0-0.2.dat')]  
r0101 =  elements[names.index('NormDensidadEscalon-1.0-1.0-0.1.dat')]  
r01010 =  elements[names.index('NormDensidadEscalon-1.0-1.0-1.0.dat')]


r1205 =  elements[names.index('NormDensidadEscalon-1.0-1.2-0.5.dat')]  
r1204 =  elements[names.index('NormDensidadEscalon-1.0-1.2-0.4.dat')] 
r1203 =  elements[names.index('NormDensidadEscalon-1.0-1.2-0.3.dat')]  
r1202 =  elements[names.index('NormDensidadEscalon-1.0-1.2-0.2.dat')]  
r1201 =  elements[names.index('NormDensidadEscalon-1.0-1.2-0.1.dat')]  
r12010 =  elements[names.index('NormDensidadEscalon-1.0-1.2-1.0.dat')]

r1505 =  elements[names.index('NormDensidadEscalon-1.0-1.5-0.5.dat')]  
r1504 =  elements[names.index('NormDensidadEscalon-1.0-1.5-0.4.dat')] 
r1503 =  elements[names.index('NormDensidadEscalon-1.0-1.5-0.3.dat')]  
r1502 =  elements[names.index('NormDensidadEscalon-1.0-1.5-0.2.dat')]  
r1501 =  elements[names.index('NormDensidadEscalon-1.0-1.5-0.1.dat')]  
r15010 =  elements[names.index('NormDensidadEscalon-1.0-1.5-1.0.dat')]


r205 =  elements[names.index('NormDensidadEscalon-1.0-2.0-0.5.dat')]  
r204 =  elements[names.index('NormDensidadEscalon-1.0-2.0-0.4.dat')] 
r203 =  elements[names.index('NormDensidadEscalon-1.0-2.0-0.3.dat')]  
r202 =  elements[names.index('NormDensidadEscalon-1.0-2.0-0.2.dat')]  
r201 =  elements[names.index('NormDensidadEscalon-1.0-2.0-0.1.dat')]  
r2010 =  elements[names.index('NormDensidadEscalon-1.0-2.0-1.0.dat')]



# # !! rho para tau = 10.0


r10015 =  elements[names.index('NormDensidadEscalon-10.0-0.1-0.5.dat')]  
r10014 =  elements[names.index('NormDensidadEscalon-10.0-0.1-0.4.dat')] 
r10013 =  elements[names.index('NormDensidadEscalon-10.0-0.1-0.3.dat')]  
r10012 =  elements[names.index('NormDensidadEscalon-10.0-0.1-0.2.dat')]  
r10011 =  elements[names.index('NormDensidadEscalon-10.0-0.1-0.1.dat')]  
r100110 =  elements[names.index('NormDensidadEscalon-10.0-0.1-1.0.dat')]  




r1002 =  elements[names.index('NormDensidadEscalon-10.0-0.2-0.4.dat')]   
r10021 =  elements[names.index('NormDensidadEscalon-10.0-0.2-0.1.dat')]   
r10022 =  elements[names.index('NormDensidadEscalon-10.0-0.2-0.2.dat')]   
r10023 =  elements[names.index('NormDensidadEscalon-10.0-0.2-0.3.dat')]   
r10025 =  elements[names.index('NormDensidadEscalon-10.0-0.2-0.5.dat')]   
r100210 =  elements[names.index('NormDensidadEscalon-10.0-0.2-1.0.dat')]  


r10045 =  elements[names.index('NormDensidadEscalon-10.0-0.4-0.5.dat')]  
r10044 =  elements[names.index('NormDensidadEscalon-10.0-0.4-0.4.dat')] 
r10043 =  elements[names.index('NormDensidadEscalon-10.0-0.4-0.3.dat')]  
r10042 =  elements[names.index('NormDensidadEscalon-10.0-0.4-0.2.dat')]  
r10041 =  elements[names.index('NormDensidadEscalon-10.0-0.4-0.1.dat')]  
r100410 =  elements[names.index('NormDensidadEscalon-10.0-0.4-1.0.dat')]


r10055 =  elements[names.index('NormDensidadEscalon-10.0-0.5-0.5.dat')]  
r10054 =  elements[names.index('NormDensidadEscalon-10.0-0.5-0.4.dat')] 
r10053 =  elements[names.index('NormDensidadEscalon-10.0-0.5-0.3.dat')]  
r10052 =  elements[names.index('NormDensidadEscalon-10.0-0.5-0.2.dat')]  
r10051 =  elements[names.index('NormDensidadEscalon-10.0-0.5-0.1.dat')]  
r100510 =  elements[names.index('NormDensidadEscalon-10.0-0.5-1.0.dat')]




r10075 =  elements[names.index('NormDensidadEscalon-10.0-0.7-0.5.dat')]  
r10074 =  elements[names.index('NormDensidadEscalon-10.0-0.7-0.4.dat')] 
r10073 =  elements[names.index('NormDensidadEscalon-10.0-0.7-0.3.dat')]  
r10072 =  elements[names.index('NormDensidadEscalon-10.0-0.7-0.2.dat')]  
r10071 =  elements[names.index('NormDensidadEscalon-10.0-0.7-0.1.dat')]  
r100710 =  elements[names.index('NormDensidadEscalon-10.0-0.7-1.0.dat')]

r100105 =  elements[names.index('NormDensidadEscalon-10.0-1.0-0.5.dat')]  
r100104 =  elements[names.index('NormDensidadEscalon-10.0-1.0-0.4.dat')] 
r100103 =  elements[names.index('NormDensidadEscalon-10.0-1.0-0.3.dat')]  
r100102 =  elements[names.index('NormDensidadEscalon-10.0-1.0-0.2.dat')]  
r100101 =  elements[names.index('NormDensidadEscalon-10.0-1.0-0.1.dat')]  
r1001010 =  elements[names.index('NormDensidadEscalon-10.0-1.0-1.0.dat')]


r101205 =  elements[names.index('NormDensidadEscalon-10.0-1.2-0.5.dat')]  
r101204 =  elements[names.index('NormDensidadEscalon-10.0-1.2-0.4.dat')] 
r101203 =  elements[names.index('NormDensidadEscalon-10.0-1.2-0.3.dat')]  
r101202 =  elements[names.index('NormDensidadEscalon-10.0-1.2-0.2.dat')]  
r101201 =  elements[names.index('NormDensidadEscalon-10.0-1.2-0.1.dat')]  
r1012010 =  elements[names.index('NormDensidadEscalon-10.0-1.2-1.0.dat')]

r101505 =  elements[names.index('NormDensidadEscalon-10.0-1.5-0.5.dat')]  
r101504 =  elements[names.index('NormDensidadEscalon-10.0-1.5-0.4.dat')] 
r101503 =  elements[names.index('NormDensidadEscalon-10.0-1.5-0.3.dat')]  
r101502 =  elements[names.index('NormDensidadEscalon-10.0-1.5-0.2.dat')]  
r101501 =  elements[names.index('NormDensidadEscalon-10.0-1.5-0.1.dat')]  
r1015010 =  elements[names.index('NormDensidadEscalon-10.0-1.5-1.0.dat')]


r10205 =  elements[names.index('NormDensidadEscalon-10.0-2.0-0.5.dat')]  
r10204 =  elements[names.index('NormDensidadEscalon-10.0-2.0-0.4.dat')] 
r10203 =  elements[names.index('NormDensidadEscalon-10.0-2.0-0.3.dat')]  
r10202 =  elements[names.index('NormDensidadEscalon-10.0-2.0-0.2.dat')]  
r10201 =  elements[names.index('NormDensidadEscalon-10.0-2.0-0.1.dat')]  
r102010 =  elements[names.index('NormDensidadEscalon-10.0-2.0-1.0.dat')]



# # !! rho para tau = 0.1


r01015 =  elements[names.index('NormDensidadEscalon-0.1-0.1-0.5.dat')]  
r01014 =  elements[names.index('NormDensidadEscalon-0.1-0.1-0.4.dat')] 
r01013 =  elements[names.index('NormDensidadEscalon-0.1-0.1-0.3.dat')]  
r01012 =  elements[names.index('NormDensidadEscalon-0.1-0.1-0.2.dat')]  
r01011 =  elements[names.index('NormDensidadEscalon-0.1-0.1-0.1.dat')]  
r010110 =  elements[names.index('NormDensidadEscalon-0.1-0.1-1.0.dat')]  




r01002 =  elements[names.index('NormDensidadEscalon-0.1-0.2-0.4.dat')]   
r010021 =  elements[names.index('NormDensidadEscalon-0.1-0.2-0.1.dat')]   
r010022 =  elements[names.index('NormDensidadEscalon-0.1-0.2-0.2.dat')]   
r010023 =  elements[names.index('NormDensidadEscalon-0.1-0.2-0.3.dat')]   
r010025 =  elements[names.index('NormDensidadEscalon-0.1-0.2-0.5.dat')]   
r0100210 =  elements[names.index('NormDensidadEscalon-0.1-0.2-1.0.dat')]  


r01045 =  elements[names.index('NormDensidadEscalon-0.1-0.4-0.5.dat')]  
r01044 =  elements[names.index('NormDensidadEscalon-0.1-0.4-0.4.dat')] 
r01043 =  elements[names.index('NormDensidadEscalon-0.1-0.4-0.3.dat')]  
r01042 =  elements[names.index('NormDensidadEscalon-0.1-0.4-0.2.dat')]  
r01041 =  elements[names.index('NormDensidadEscalon-0.1-0.4-0.1.dat')]  
r010410 =  elements[names.index('NormDensidadEscalon-0.1-0.4-1.0.dat')]


r01055 =  elements[names.index('NormDensidadEscalon-0.1-0.5-0.5.dat')]  
r01054 =  elements[names.index('NormDensidadEscalon-0.1-0.5-0.4.dat')] 
r01053 =  elements[names.index('NormDensidadEscalon-0.1-0.5-0.3.dat')]  
r01052 =  elements[names.index('NormDensidadEscalon-0.1-0.5-0.2.dat')]  
r01051 =  elements[names.index('NormDensidadEscalon-0.1-0.5-0.1.dat')]  
r010510 =  elements[names.index('NormDensidadEscalon-0.1-0.5-1.0.dat')]




r01075 =  elements[names.index('NormDensidadEscalon-0.1-0.7-0.5.dat')]  
r01074 =  elements[names.index('NormDensidadEscalon-0.1-0.7-0.4.dat')] 
r01073 =  elements[names.index('NormDensidadEscalon-0.1-0.7-0.3.dat')]  
r01072 =  elements[names.index('NormDensidadEscalon-0.1-0.7-0.2.dat')]  
r01071 =  elements[names.index('NormDensidadEscalon-0.1-0.7-0.1.dat')]  
r010710 =  elements[names.index('NormDensidadEscalon-0.1-0.7-1.0.dat')]

r010105 =  elements[names.index('NormDensidadEscalon-0.1-1.0-0.5.dat')]  
r010104 =  elements[names.index('NormDensidadEscalon-0.1-1.0-0.4.dat')] 
r010103 =  elements[names.index('NormDensidadEscalon-0.1-1.0-0.3.dat')]  
r010102 =  elements[names.index('NormDensidadEscalon-0.1-1.0-0.2.dat')]  
r010101 =  elements[names.index('NormDensidadEscalon-0.1-1.0-0.1.dat')]  
r0101010 =  elements[names.index('NormDensidadEscalon-0.1-1.0-1.0.dat')]


r011205 =  elements[names.index('NormDensidadEscalon-0.1-1.2-0.5.dat')]  
r011204 =  elements[names.index('NormDensidadEscalon-0.1-1.2-0.4.dat')] 
r011203 =  elements[names.index('NormDensidadEscalon-0.1-1.2-0.3.dat')]  
r011202 =  elements[names.index('NormDensidadEscalon-0.1-1.2-0.2.dat')]  
r011201 =  elements[names.index('NormDensidadEscalon-0.1-1.2-0.1.dat')]  
r0112010 =  elements[names.index('NormDensidadEscalon-0.1-1.2-1.0.dat')]

r011505 =  elements[names.index('NormDensidadEscalon-0.1-1.5-0.5.dat')]  
r011504 =  elements[names.index('NormDensidadEscalon-0.1-1.5-0.4.dat')] 
r011503 =  elements[names.index('NormDensidadEscalon-0.1-1.5-0.3.dat')]  
r011502 =  elements[names.index('NormDensidadEscalon-0.1-1.5-0.2.dat')]  
r011501 =  elements[names.index('NormDensidadEscalon-0.1-1.5-0.1.dat')]  
r0115010 =  elements[names.index('NormDensidadEscalon-0.1-1.5-1.0.dat')]


r01205 =  elements[names.index('NormDensidadEscalon-0.1-2.0-0.5.dat')]  
r01204 =  elements[names.index('NormDensidadEscalon-0.1-2.0-0.4.dat')] 
r01203 =  elements[names.index('NormDensidadEscalon-0.1-2.0-0.3.dat')]  
r01202 =  elements[names.index('NormDensidadEscalon-0.1-2.0-0.2.dat')]  
r01201 =  elements[names.index('NormDensidadEscalon-0.1-2.0-0.1.dat')]  
r012010 =  elements[names.index('NormDensidadEscalon-0.1-2.0-1.0.dat')]



ro1= pd.read_csv("density_lambda1.dat",header=None,sep='\s+',names=["pos","dens"])
ro02= pd.read_table("density_lambda0.2_.dat",header=None,sep='\s+',names=["pos","dens","error"])
ro04= pd.read_table("density_lambda0.4_.dat",header=None,sep='\s+',names=["pos","dens","error"])
ro05= pd.read_csv("density_lamb0.5_1.dat",header=None,sep='\s+',names=["pos","dens","errror"])
ro07= pd.read_table("density_lambda0.7_.dat",header=None,sep='\s+',names=["pos","dens","error"])
ro12= pd.read_table("density_lambda1.2_.dat",header=None,sep='\s+',names=["pos","dens"])
ro15= pd.read_table("density_lambda1.5_.dat",header=None,sep='\s+',names=["pos","dens"])






N = 1000 # 2000

# Parametros del sistema

v0 = V = B = D =  1
alpha = 0.5
l0=0.4

############## Valores gradiente y tiempo de memoria

Tm = 1.0

##############

L = 20.0

dx = L/300

# Arreglo espacial de la configuracion

xfield = np.arange(0,L,dx)

x_bin = xfield + .5*dx


xx = np.arange(-L/2,L/2,dx)


# !!!!!!!!!!!! FUNCIONES !!!
def ligando_escalar(x,l0):
    
#    return math.tanh(4*math.sin(2*math.pi*x/L))

    if(x<L/2):

        return -l0

    else:

        return l0


def lorentzian_form(lamb,tau,x):
    
   psi0=lamb*np.exp(lamb**2/2)/(1+np.exp(lamb**2/2)*tau*(1+lamb**2))
   k0 = np.sqrt((1+np.exp(lamb**2/2)*tau*(1+lamb**2))/tau**2)
   
   return  psi0*(1 - np.exp(- k0* abs(x)))/k0

def mu0_2(lamb,tau): 
    return (lamb * (2 + np.exp(lamb**2/2) * (1 + lamb**2) *  tau))/(2 + np.exp(lamb**2/2) *  (3 + 2  * lamb**2) *  tau + np.exp(lamb**2) * tau**2)
def D_2(lamb,tau):
    return   (np.exp(-(lamb**2/2)) *  (4 + np.exp(lamb**2/2) *  (6 + 8 *  lamb**2 + lamb**4) *  tau + np.exp(lamb**2) *  (2 + 2 * lamb**2 + lamb**4) *  tau**2))/(2  * (2 + np.exp(lamb**2/2) *  (3 + 2 * lamb**2) * tau + np.exp(lamb**2) *  tau**2))
def Dx_2(lamb,tau):
    return   (lamb*  tau *  (2 + np.exp(lamb**2/2)  * (1 + lamb**2) *  tau))/(2 + np.exp(lamb**2/2) *  (3 + 2 * lamb**2) *  tau + np.exp(lamb**2) * tau**2)
def DDx_2(lamb,tau):
    return ( tau *  (2 + np.exp(lamb**2/2)  * (1 + lamb**2) *  tau))/(2 + np.exp(lamb**2/2) *  (3 + 2 * lamb**2) *  tau + np.exp(lamb**2) * tau**2)
def psi0(lamb,tau):
    return mu0_2(lamb,tau)/ D_2(lamb,tau)
def k0_2(lamb,tau):
    return np.sqrt(D_2(lamb,tau)/(tau*(D_2(lamb,tau)*DDx_2(lamb,tau)-Dx_2(lamb,tau)**2)))

def k0(lamb,tau):
    return np.sqrt((1+np.exp(lamb**2/2)*tau*(1+lamb**2))/tau**2)

def lorentzian_form_order_2(lamb,tau,x):
  
   
   
   return  psi0(lamb,tau)*(1 - np.exp(- k0_2(lamb,tau)* abs(x)))





def lorentzian_form_articulo(lamb,tau,x):
    
   return lamb*tau*(1 - np.exp(-np.sqrt(1+tau)*abs(x)/tau))/(1+tau)


ligando = np.vectorize(ligando_escalar)


def lorentz(x,a,b):
   return  a*np.sign(x)*(1 - np.exp(- b* abs(x)))

def ro_prime(vector):
    
    i=0
    ro_prima=np.zeros(int(len(xx)/4))
    for x in (vector['pos']):
        if x<L/4:
        
            ro_prima[i] = 0.5*( vector['dens'][list(vector['pos']).index(x)] + vector['dens'][list(vector['pos']).index(find_nearest(vector['pos'], L/2-x))] - vector['dens'][list(vector['pos']).index(find_nearest(vector['pos'], x+3*L/2))] + vector['dens'][list(vector['pos']).index(find_nearest(vector['pos'],L- x))]  )

        i+=1
       
    return ro_prima
def ro_pp(vector):
    
    i=0
    ro_pr=np.zeros([int(len(vector[0])/2),1])
    
    for x in (vector[0]):
        # print(x)
        if  abs(x)<L/4:
            # print(i)
            ro_pr[i] =float(vector.loc[vector[0] == x][1])
            # ro_pr[i][0] =x
            # print(ro_pr)
            i+=1
       
    return ro_pr

def l_c_teorica(lamda,tau):
    return tau/np.sqrt(1+np.exp(lamda**2/2)*tau*(1+lamda**2) )


def calculo_k0_simulaciones(vector1,vector2,vector3,vector4,vector5,vector6):
    
    r4_p = ro_pp(vector1[0:])

    xxx = np.linspace(-L/4,L/4,len(r4_p))

    popt4, pcov4 = curve_fit(lorentz, xxx, np.reshape(r4_p, r4_p.size))


    r5_p = ro_pp(vector2[0:])
    popt5, pcov5 = curve_fit(lorentz, xxx, np.reshape(r5_p, r5_p.size))

    r3_p = ro_pp(vector3[0:])
    popt3, pcov3 = curve_fit(lorentz, xxx, np.reshape(r3_p, r3_p.size))

    r2_p = ro_pp(vector4[0:])
    popt2, pcov2 = curve_fit(lorentz, xxx, np.reshape(r2_p, r2_p.size))

    r10_p = ro_pp(vector5[0:])
    popt10, pcov10 = curve_fit(lorentz, xxx, np.reshape(r10_p, r10_p.size))


    r1_p = ro_pp(vector6[0:])
    
    popt1, pcov1 = curve_fit(lorentz, xxx, np.reshape(r1_p, r1_p.size))



    k010_media= (popt5[1]+popt3[1]+popt2[1]+popt1[1]+popt4[1]+popt10[1])/6
    
    psi010_media= (popt5[0]+popt3[0]+popt2[0]+popt1[0]+popt4[0]+popt10[0])/6



    sd_k010 = np.sqrt((popt5[1]**2+popt3[1]**2+popt2[1]**2+popt1[1]**2+popt4[1]**2+popt10[1]**2)/6-k010_media**2)

    
    return [k010_media,sd_k010,psi010_media]

# !! !!!!!!!!!!!!!!!!!!!!!!!!!!  #
# print(r014[0:])

# r1_p =ro_pp(r015[0:])



# r1_p.append(ro_pp(r0110[0:]))


# r1_p.append(ro_pp(r012[0:]))


# r1_p.concat(ro_pp(r013[0:]))


# !!! calculo de K0 

# !! K0 para  lambda = 1 

r10_p = ro_pp(r0104[0:])

xxx = np.linspace(-L/4,L/4,len(r10_p))

popt1004, pcov1004 = curve_fit(lorentz, xxx, np.reshape(r10_p, r10_p.size))


r105_p = ro_pp(r0105[0:])
popt1005, pcov1005 = curve_fit(lorentz, xxx, np.reshape(r105_p, r105_p.size))

r103_p = ro_pp(r0103[0:])
popt1003, pcov1003 = curve_fit(lorentz, xxx, np.reshape(r103_p, r103_p.size))

r102_p = ro_pp(r0102[0:])
popt1002, pcov1002 = curve_fit(lorentz, xxx, np.reshape(r102_p, r102_p.size))

r1010_p = ro_pp(r0101[0:])
popt10010, pcov10010 = curve_fit(lorentz, xxx, np.reshape(r1010_p, r1010_p.size))


r101_p = ro_pp(r01010[0:])
popt1001, pcov1001 = curve_fit(lorentz, xxx, np.reshape(r101_p, r101_p.size))



k010_media= (popt1005[1]+popt1003[1]+popt1002[1]+popt1001[1]+popt1004[1]+popt10010[1])/6



sd_k010 = np.sqrt((popt1005[1]**2+popt1003[1]**2+popt1002[1]**2+popt1001[1]**2+popt1004[1]**2+popt10010[1]**2)/6-k010_media**2)

error_k010= (np.diag(pcov1003)+np.diag(pcov1005)+np.diag(pcov1002)+np.diag(pcov1001)+np.diag(pcov1004)+np.diag(pcov10010))

# !! K0  con la función  para tau = 1.0

k010_media_2 = calculo_k0_simulaciones(r0104,r0105,r0103,r0102,r0101,r01010)
k002_media_2 = calculo_k0_simulaciones(r025,r02,r023,r022,r0210,r021)
k001_media_2 = calculo_k0_simulaciones(r015,r014,r013,r012,r0110,r011)
k004_media_2 = calculo_k0_simulaciones(r045,r044,r043,r042,r0410,r041)
k005_media_2 = calculo_k0_simulaciones(r055,r054,r053,r052,r0510,r051)
k007_media_2 = calculo_k0_simulaciones(r075,r074,r073,r072,r0710,r071)
k0012_media_2 = calculo_k0_simulaciones(r1205,r1204,r1203,r1202,r12010,r1201)
k0015_media_2 = calculo_k0_simulaciones(r1505,r1504,r1503,r1502,r15010,r1501)
k0020_media_2 = calculo_k0_simulaciones(r205,r204,r203,r202,r2010,r201)


# !! K0  con la función  para tau = 10.0

k01001_media_2 = calculo_k0_simulaciones(r100104,r100105,r100103,r100102,r100101,r1001010)
k01002_media_2 = calculo_k0_simulaciones(r10025,r1002,r10023,r10022,r100210,r10021)
k01010_media_2 = calculo_k0_simulaciones(r10015,r10014,r10013,r10012,r100110,r10011)
k01004_media_2 = calculo_k0_simulaciones(r10045,r10044,r10043,r10042,r100410,r10041)
k01005_media_2 = calculo_k0_simulaciones(r10055,r10054,r10053,r10052,r100510,r10051)
k01007_media_2 = calculo_k0_simulaciones(r10075,r10074,r10073,r10072,r100710,r10071)
k010012_media_2 = calculo_k0_simulaciones(r101205,r101204,r101203,r101202,r1012010,r101201)
k010015_media_2 = calculo_k0_simulaciones(r101505,r101504,r101503,r101502,r1015010,r101501)
k010020_media_2 = calculo_k0_simulaciones(r10205,r10204,r10203,r10202,r102010,r10201)


# !! K0  con la función  para tau = 0.1

k00110_media_2 = calculo_k0_simulaciones(r01015,r01014,r01013,r01012,r010110,r01011)
k00102_media_2 = calculo_k0_simulaciones(r01002,r010021,r010022,r010023,r010025,r0100210)
k00101_media_2 = calculo_k0_simulaciones(r010104,r010105,r010103,r010102,r010101,r0101010) 
k00104_media_2 = calculo_k0_simulaciones(r01045,r01044,r01043,r01042,r010410,r01041)
k00105_media_2 = calculo_k0_simulaciones(r01055,r01054,r01053,r01052,r010510,r01051)
k00107_media_2 = calculo_k0_simulaciones(r01075,r01074,r01073,r01072,r010710,r01071)
k001012_media_2 = calculo_k0_simulaciones(r011205,r011204,r011203,r011202,r0112010,r011201)
k001015_media_2 = calculo_k0_simulaciones(r011505,r011504,r011503,r011502,r0115010,r011501)
k001020_media_2 = calculo_k0_simulaciones(r01205,r01204,r01203,r01202,r012010,r01201)




# print("error k0 para lambda=1.0",pcov105+pcov103+popt102+pcov101+pcov1004)
# print("error k0 para lambda=1.0",error_k010)
# plt.plot(xxx, lorentz(xxx,popt1004[0],popt1004[1]))
# !! K0 para  lambda = 0.2

r205_p = ro_pp(r025[0:])
popt205, pcov205 = curve_fit(lorentz, xxx, np.reshape(r205_p, r205_p.size))

r204_p = ro_pp(r02[0:])
popt204, pcov204 = curve_fit(lorentz, xxx, np.reshape(r204_p, r204_p.size))


r203_p = ro_pp(r023[0:])
popt203, pcov203 = curve_fit(lorentz, xxx, np.reshape(r203_p, r203_p.size))

r202_p = ro_pp(r022[0:])
popt202, pcov202 = curve_fit(lorentz, xxx, np.reshape(r202_p, r202_p.size))

r2010_p = ro_pp(r0210[0:])
popt2010, pcov2010 = curve_fit(lorentz, xxx, np.reshape(r2010_p, r2010_p.size))

r201_p = ro_pp(r021[0:])
popt201, pcov201 = curve_fit(lorentz, xxx, np.reshape(r201_p, r201_p.size))



k002_media= (popt205[1]+popt203[1]+popt202[1]+popt201[1]+popt204[1]+popt2010[1])/6


sd_k002 = np.sqrt((popt205[1]**2+popt203[1]**2+popt202[1]**2+popt201[1]**2+popt204[1]**2+popt2010[1]**2)/6-k002_media**2)


error_k002= (np.diag(pcov203)+np.diag(pcov205)+np.diag(pcov202)+np.diag(pcov201)+np.diag(pcov204)+np.diag(pcov2010))


# !! K0 para  lambda = 0.4

r405_p = ro_pp(r045[0:])
popt405, pcov405 = curve_fit(lorentz, xxx, np.reshape(r405_p, r405_p.size))

r404_p = ro_pp(r044[0:])
popt404, pcov404 = curve_fit(lorentz, xxx, np.reshape(r404_p, r404_p.size))


r403_p = ro_pp(r043[0:])
popt403, pcov403 = curve_fit(lorentz, xxx, np.reshape(r403_p, r403_p.size))

r402_p = ro_pp(r042[0:])
popt402, pcov402 = curve_fit(lorentz, xxx, np.reshape(r402_p, r402_p.size))

r4010_p = ro_pp(r0410[0:])
popt4010, pcov4010 = curve_fit(lorentz, xxx, np.reshape(r4010_p, r4010_p.size))


r401_p = ro_pp(r041[0:])
popt401, pcov401 = curve_fit(lorentz, xxx, np.reshape(r401_p, r401_p.size))


k04_media= (popt405[1]+popt403[1]+popt402[1]+popt401[1]+popt404[1]+popt4010[1])/6


sd_k04 = np.sqrt((popt405[1]**2+popt403[1]**2+popt402[1]**2+popt401[1]**2+popt404[1]**2+popt4010[1]**2)/6-k04_media**2)


error_k04= (np.diag(pcov403)+np.diag(pcov405)+np.diag(pcov402)+np.diag(pcov401)+np.diag(pcov404)+np.diag(pcov4010))



# !! K0 para  lambda = 0.1


r105_p = ro_pp(r015[0:])
popt105, pcov105 = curve_fit(lorentz, xxx, np.reshape(r105_p, r105_p.size))

r104_p = ro_pp(r014[0:])
popt104, pcov104 = curve_fit(lorentz, xxx, np.reshape(r104_p, r104_p.size))


r103_p = ro_pp(r013[0:])
popt103, pcov103 = curve_fit(lorentz, xxx, np.reshape(r103_p, r103_p.size))

r102_p = ro_pp(r012[0:])
popt102, pcov102 = curve_fit(lorentz, xxx, np.reshape(r102_p, r102_p.size))

r1010_p = ro_pp(r0110[0:])
popt1010, pcov1010 = curve_fit(lorentz, xxx, np.reshape(r1010_p, r1010_p.size))

r101_p = ro_pp(r011[0:])
popt101, pcov101 = curve_fit(lorentz, xxx, np.reshape(r101_p, r101_p.size))



k01_media= (popt105[1]+popt103[1]+popt102[1]+popt1010[1]+popt104[1]+popt101[1])/6

sd_k01 = np.sqrt((popt105[1]**2+popt103[1]**2+popt102[1]**2+popt101[1]**2+popt104[1]**2+popt1010[1]**2)/6-k01_media**2)



error_k01= (np.diag(pcov103)+np.diag(pcov105)+np.diag(pcov102)+np.diag(pcov101)+np.diag(pcov104)+np.diag(pcov101))




# !! K0 para  lambda = 0.5


r505_p = ro_pp(r055[0:])
popt505, pcov505 = curve_fit(lorentz, xxx, np.reshape(r505_p, r505_p.size))

r504_p = ro_pp(r054[0:])
popt504, pcov504 = curve_fit(lorentz, xxx, np.reshape(r504_p, r504_p.size))


r503_p = ro_pp(r053[0:])
popt503, pcov503 = curve_fit(lorentz, xxx, np.reshape(r503_p, r503_p.size))

r502_p = ro_pp(r052[0:])
popt502, pcov502 = curve_fit(lorentz, xxx, np.reshape(r502_p, r502_p.size))

r5010_p = ro_pp(r0510[0:])
popt5010, pcov5010 = curve_fit(lorentz, xxx, np.reshape(r5010_p, r5010_p.size))

r501_p = ro_pp(r051[0:])
popt501, pcov501 = curve_fit(lorentz, xxx, np.reshape(r501_p, r501_p.size))



k05_media= (popt505[1]+popt503[1]+popt502[1]+popt5010[1]+popt504[1]+popt501[1])/6


sd_k05 = np.sqrt((popt505[1]**2+popt503[1]**2+popt502[1]**2+popt501[1]**2+popt504[1]**2+popt5010[1]**2)/6-k05_media**2)



error_k05= (np.diag(pcov503)+np.diag(pcov505)+np.diag(pcov502)+np.diag(pcov501)+np.diag(pcov504)+np.diag(pcov501))


# !! K0 para  lambda = 0.7


r705_p = ro_pp(r075[0:])
popt705, pcov705 = curve_fit(lorentz, xxx, np.reshape(r705_p, r705_p.size))

r704_p = ro_pp(r074[0:])
popt704, pcov704 = curve_fit(lorentz, xxx, np.reshape(r704_p, r704_p.size))


r703_p = ro_pp(r073[0:])
popt703, pcov703 = curve_fit(lorentz, xxx, np.reshape(r703_p, r703_p.size))

r702_p = ro_pp(r072[0:])
popt702, pcov702 = curve_fit(lorentz, xxx, np.reshape(r702_p, r702_p.size))

r7010_p = ro_pp(r0710[0:])
popt7010, pcov7010 = curve_fit(lorentz, xxx, np.reshape(r7010_p, r7010_p.size))

r701_p = ro_pp(r071[0:])
popt701, pcov701 = curve_fit(lorentz, xxx, np.reshape(r701_p, r701_p.size))



k07_media= (popt705[1]+popt703[1]+popt702[1]+popt7010[1]+popt704[1]+popt701[1])/6


sd_k07 = np.sqrt((popt705[1]**2+popt703[1]**2+popt702[1]**2+popt701[1]**2+popt704[1]**2+popt7010[1]**2)/6-k07_media**2)



error_k07= (np.diag(pcov703)+np.diag(pcov705)+np.diag(pcov702)+np.diag(pcov701)+np.diag(pcov704)+np.diag(pcov701))


# !! K0 para  lambda = 1.2

r1205_p = ro_pp(r1205[0:])
popt1205, pcov1205 = curve_fit(lorentz, xxx, np.reshape(r1205_p, r1205_p.size))

r1204_p = ro_pp(r1204[0:])
popt1204, pcov1204 = curve_fit(lorentz, xxx, np.reshape(r1204_p, r1204_p.size))


r1203_p = ro_pp(r1203[0:])
popt1203, pcov1203 = curve_fit(lorentz, xxx, np.reshape(r1203_p, r1203_p.size))

r1202_p = ro_pp(r1202[0:])
popt1202, pcov1202 = curve_fit(lorentz, xxx, np.reshape(r1202_p, r1202_p.size))

r12010_p = ro_pp(r12010[0:])
popt12010, pcov12010 = curve_fit(lorentz, xxx, np.reshape(r12010_p, r12010_p.size))

r1201_p = ro_pp(r1201[0:])
popt1201, pcov1201 = curve_fit(lorentz, xxx, np.reshape(r1201_p, r1201_p.size))



k012_media= (popt1205[1]+popt1203[1]+popt1202[1]+popt12010[1]+popt1204[1]+popt1201[1])/6

sd_k012 = np.sqrt((popt1205[1]**2+popt1203[1]**2+popt1202[1]**2+popt1201[1]**2+popt1204[1]**2+popt12010[1]**2)/6-k012_media**2)

error_k012= (np.diag(pcov1203)+np.diag(pcov1205)+np.diag(pcov1202)+np.diag(pcov1201)+np.diag(pcov1204)+np.diag(pcov1201))

# !! K0 para  lambda = 1.5

r1505_p = ro_pp(r1505[0:])

popt1505, pcov1505 = curve_fit(lorentz, xxx, np.reshape(r1505_p, r1505_p.size))

r1504_p = ro_pp(r1504[0:])

popt1504, pcov1504 = curve_fit(lorentz, xxx, np.reshape(r1504_p, r1504_p.size))

r1503_p = ro_pp(r1503[0:])

popt1503, pcov1503 = curve_fit(lorentz, xxx, np.reshape(r1503_p, r1503_p.size))

r1502_p = ro_pp(r1502[0:])

popt1502, pcov1502 = curve_fit(lorentz, xxx, np.reshape(r1502_p, r1502_p.size))

r15010_p = ro_pp(r15010[0:])

popt15010, pcov15010 = curve_fit(lorentz, xxx, np.reshape(r15010_p, r15010_p.size))

r1501_p = ro_pp(r1501[0:])

popt1501, pcov1501 = curve_fit(lorentz, xxx, np.reshape(r1501_p, r1501_p.size))

k015_media= (popt1505[1]+popt1503[1]+popt1502[1]+popt15010[1]+popt1504[1]+popt1501[1])/6

sd_k015 = np.sqrt((popt1505[1]**2+popt1503[1]**2+popt1502[1]**2+popt1501[1]**2+popt1504[1]**2+popt15010[1]**2)/6-k015_media**2)

error_k015= (np.diag(pcov1503)+np.diag(pcov1505)+np.diag(pcov1502)+np.diag(pcov1501)+np.diag(pcov1504)+np.diag(pcov1501))

# !! K0 para  lambda = 2.0

r205_p = ro_pp(r205[0:])

popt2005, pcov2005 = curve_fit(lorentz, xxx, np.reshape(r205_p, r205_p.size))

r204_p = ro_pp(r204[0:])

popt2004, pcov2004 = curve_fit(lorentz, xxx, np.reshape(r204_p, r204_p.size))

r203_p = ro_pp(r203[0:])

popt2003, pcov2003 = curve_fit(lorentz, xxx, np.reshape(r203_p, r203_p.size))

r202_p = ro_pp(r202[0:])

popt2002, pcov2002 = curve_fit(lorentz, xxx, np.reshape(r202_p, r202_p.size))

r2010_p = ro_pp(r2010[0:])

popt20010, pcov20010 = curve_fit(lorentz, xxx, np.reshape(r2010_p, r2010_p.size))

r201_p = ro_pp(r201[0:])

popt2001, pcov2001 = curve_fit(lorentz, xxx, np.reshape(r201_p, r201_p.size))

k02_media= (popt2005[1]+popt2003[1]+popt2002[1]+popt20010[1]+popt2004[1]+popt2001[1])/6

sd_k02 = np.sqrt((popt2005[1]**2+popt2003[1]**2+popt2002[1]**2+popt2001[1]**2+popt2004[1]**2+popt20010[1]**2)/6-k02_media**2)

error_k02= (np.diag(pcov2003)+np.diag(pcov2005)+np.diag(pcov2002)+np.diag(pcov2001)+np.diag(pcov2004)+np.diag(pcov2001))

r02_p = ro_pp(r02[0:])

popt02, pcov02 = curve_fit(lorentz, xxx, np.reshape(r02_p, r02_p.size))

print(popt02[0])

print('error lambda=0.2',(1/popt02[0])**2*np.diag(pcov02)[1])

r044_p = ro_pp(r044[0:])

popt04, pcov04 = curve_fit(lorentz, xxx, np.reshape(r044_p, r044_p.size))

print(popt04[0])

print('error lambda=1',(1/popt04[0])**2*np.diag(pcov04)[1])

r054_p = ro_pp(r054[0:])

popt05, pcov05 = curve_fit(lorentz, xxx, np.reshape(r054_p, r054_p.size))

print(popt05)

r074_p = ro_pp(r074[0:])

popt07, pcov07 = curve_fit(lorentz, xxx, np.reshape(r074_p, r074_p.size))

print(popt07)

r014_p = ro_pp(r014[0:])

popt01, pcov01 = curve_fit(lorentz, xxx, np.reshape(r014_p, r014_p.size))

r012_p = ro_pp(r012[0:])

popt012, pcov012 = curve_fit(lorentz, xxx, np.reshape(r012_p, r012_p.size))

r013_p = ro_pp(r013[0:])

popt013, pcov013 = curve_fit(lorentz, xxx, np.reshape(r013_p, r013_p.size))
 
r015_p = ro_pp(r015[0:])



popt015, pcov015 = curve_fit(lorentz, xxx, np.reshape(r015_p, r015_p.size))

r0110_p = ro_pp(r0110[0:])


popt0110, pcov0110 = curve_fit(lorentz, xxx, np.reshape(r0110_p, r0110_p.size))
    
r12_p = ro_pp(r1204[0:])


popt12, pcov12 = curve_fit(lorentz, xxx, np.reshape(r12_p, r12_p.size))

print('error lambda=1',(1/popt12[0])**2*np.diag(pcov12)[1])
    
r15_p = ro_pp(r1504[0:])

popt15, pcov15 = curve_fit(lorentz, xxx, np.reshape(r15_p, r15_p.size))
    
r2_p = ro_pp(r204[0:])


popt2, pcov2 = curve_fit(lorentz, xxx, np.reshape(r2_p, r2_p.size))




# ????? FIGURAS para lambda=0.1 tau = 1

#! r015 =  elements[names.index('NormDensidadEscalon-1.0-0.1-0.5.dat')]  
#! r014 =  elements[names.index('NormDensidadEscalon-1.0-0.1-0.4.dat')] 
#! r013 =  elements[names.index('NormDensidadEscalon-1.0-0.1-0.3.dat')]  
#! r012 =  elements[names.index('NormDensidadEscalon-1.0-0.1-0.2.dat')]  
#! r011 =  elements[names.index('NormDensidadEscalon-1.0-0.1-0.1.dat')]  
#! r0110 =  elements[names.index('NormDensidadEscalon-1.0-0.1-1.0.dat')]  

fig1=plt.figure(figsize=(14,8))

plt.plot(r015[0:][0],r015[0:][1],color= "C0",label=r"$\lambda = 0.1,\tau=1,l_0=0.5$")

plt.plot(xx,np.sqrt(2/np.pi)*lorentzian_form(0.1,Tm,xx)*ligando(x_bin,0.5),color="C1")

plt.xlabel(r'$x$',fontsize=30)

plt.ylabel(r'$\Delta\rho/\rho_0$',fontsize=30)

plt.legend(fontsize=30)

plt.xticks(fontsize=40)
plt.yticks(fontsize=40)



plt.tight_layout()

plt.savefig("rho_lambda01_tau_1_l0_05.pdf",dpi=1200)

fig1=plt.figure(figsize=(14,8))

plt.plot(r014[0:][0],r014[0:][1],color= "C0",label=r"$\lambda = 0.1,\tau=1,l_0=0.4$")

plt.plot(xx,np.sqrt(2/np.pi)*lorentzian_form(0.1,Tm,xx)*ligando(x_bin,0.4),color="C1")

plt.xlabel(r'$x$',fontsize=30)

plt.ylabel(r'$\Delta\rho/\rho_0$',fontsize=30)

plt.legend(fontsize=30)

plt.xticks(fontsize=40)
plt.yticks(fontsize=40)



plt.tight_layout()

plt.savefig("rho_lambda01_tau_1_l0_04.pdf",dpi=1200)

fig1=plt.figure(figsize=(14,8))

plt.plot(r013[0:][0],r013[0:][1],color= "C0",label=r"$\lambda = 0.1,\tau=1,l_0=0.3$")

plt.plot(xx,np.sqrt(2/np.pi)*lorentzian_form(0.1,Tm,xx)*ligando(x_bin,0.3),color="C1")

plt.xlabel(r'$x$',fontsize=30)

plt.ylabel(r'$\Delta\rho/\rho_0$',fontsize=30)

plt.legend(fontsize=30)

plt.xticks(fontsize=40)
plt.yticks(fontsize=40)



plt.tight_layout()

plt.savefig("rho_lambda01_tau_1_l0_03.pdf",dpi=1200)

fig1=plt.figure(figsize=(14,8))

plt.plot(r012[0:][0],r012[0:][1],color= "C0",label=r"$\lambda = 0.1,\tau=1,l_0=0.2$")

plt.plot(xx,np.sqrt(2/np.pi)*lorentzian_form(0.1,Tm,xx)*ligando(x_bin,0.2),color="C1")

plt.xlabel(r'$x$',fontsize=30)

plt.ylabel(r'$\Delta\rho/\rho_0$',fontsize=30)

plt.legend(fontsize=30)

plt.xticks(fontsize=40)
plt.yticks(fontsize=40)



plt.tight_layout()

plt.savefig("rho_lambda01_tau_1_l0_02.pdf",dpi=1200)


fig1=plt.figure(figsize=(14,8))

plt.plot(r011[0:][0],r011[0:][1],color= "C0",label=r"$\lambda = 0.1,\tau=1,l_0=0.1$")



plt.plot(xx,np.sqrt(2/np.pi)*lorentzian_form(0.1,Tm,xx)*ligando(x_bin,0.1),color="C1")
# plt.plot(xx,0.49*lorentzian_form_order_2(1,Tm,xx)*ligando(x_bin,l0),color="C1",linestyle="--")

# plt.plot(xx,0.082*ligando(x_bin,l0)/0.4,color="C1",linestyle="--")
# plt.plot(xx,0.5*lorentzian_form_articulo(1,Tm,xx)*ligando(x_bin,l0),color="C1",linestyle="--")
# plt.plot(xxx,lorentz(xxx,popt1004[0],popt1004[1]),color="C1",linestyle="dashdot")


plt.xlabel(r'$x$',fontsize=30)

plt.ylabel(r'$\Delta\rho/\rho_0$',fontsize=30)

plt.legend(fontsize=30)

plt.xticks(fontsize=40)
plt.yticks(fontsize=40)



plt.tight_layout()

plt.savefig("rho_lambda1_tau_1_l0_01.pdf",dpi=1200)
fig1=plt.figure(figsize=(14,8))


plt.plot(r0110[0:][0],r0110[0:][1],color= "C0",label=r"$\lambda = 0.1,\tau=1,l_0=1.0$")



plt.plot(xx,np.sqrt(2/np.pi)*lorentzian_form(0.1,Tm,xx)*ligando(x_bin,1.0),color="C1")
# plt.plot(xx,0.49*lorentzian_form_order_2(1,Tm,xx)*ligando(x_bin,l0),color="C1",linestyle="--")

# plt.plot(xx,0.082*ligando(x_bin,l0)/0.4,color="C1",linestyle="--")
# plt.plot(xx,0.5*lorentzian_form_articulo(1,Tm,xx)*ligando(x_bin,l0),color="C1",linestyle="--")
# plt.plot(xxx,lorentz(xxx,popt1004[0],popt1004[1]),color="C1",linestyle="dashdot")


plt.xlabel(r'$x$',fontsize=30)

plt.ylabel(r'$\Delta\rho/\rho_0$',fontsize=30)

plt.legend(fontsize=30)

plt.xticks(fontsize=40)
plt.yticks(fontsize=40)



plt.tight_layout()

plt.savefig("rho_lambda1_tau_1_l0_1.pdf",dpi=1200)



# ????? FIGURAS para lambda=0.1 tau = 10.0

#! r10015 =  elements[names.index('NormDensidadEscalon-10.0-0.1-0.5.dat')]  
#! r10014 =  elements[names.index('NormDensidadEscalon-10.0-0.1-0.4.dat')] 
#! r10013 =  elements[names.index('NormDensidadEscalon-10.0-0.1-0.3.dat')]  
#! r10012 =  elements[names.index('NormDensidadEscalon-10.0-0.1-0.2.dat')]  
#! r10011 =  elements[names.index('NormDensidadEscalon-10.0-0.1-0.1.dat')]  
#! r100110 =  elements[names.index('NormDensidadEscalon-10.0-0.1-1.0.dat')]  

tau2 = 10.0
lamda2 =0.1


fig1=plt.figure(figsize=(14,8))

plt.plot(r10015[0:][0],r10015[0:][1],color= "C0",label=r"$\lambda = 0.1,\tau=10,l_0=0.5$")

plt.plot(xx,np.sqrt(np.pi)*lorentzian_form(lamda2,tau2,xx)*ligando(x_bin,0.5),color="C1")

plt.xlabel(r'$x$',fontsize=30)

plt.ylabel(r'$\Delta\rho/\rho_0$',fontsize=30)

plt.legend(fontsize=30)

plt.xticks(fontsize=40)
plt.yticks(fontsize=40)



plt.tight_layout()

plt.savefig("rho_lambda01_tau_10_l0_05.pdf",dpi=1200)

fig1=plt.figure(figsize=(14,8))

plt.plot(r10014[0:][0],r10014[0:][1],color= "C0",label=r"$\lambda = 0.1,\tau=10,l_0=0.5$")

plt.plot(xx,np.sqrt(np.pi)*lorentzian_form(lamda2,tau2,xx)*ligando(x_bin,0.4),color="C1")

plt.xlabel(r'$x$',fontsize=30)

plt.ylabel(r'$\Delta\rho/\rho_0$',fontsize=30)

plt.legend(fontsize=30)

plt.xticks(fontsize=40)
plt.yticks(fontsize=40)



plt.tight_layout()

plt.savefig("rho_lambda01_tau_10_l0_04.pdf",dpi=1200)





fig1=plt.figure(figsize=(14,8))

plt.plot(r10013[0:][0],r10013[0:][1],color= "C0",label=r"$\lambda = 0.1,\tau=10,l_0=0.3$")

plt.plot(xx,np.sqrt(np.pi)*lorentzian_form(lamda2,tau2,xx)*ligando(x_bin,0.3),color="C1")

plt.xlabel(r'$x$',fontsize=30)

plt.ylabel(r'$\Delta\rho/\rho_0$',fontsize=30)

plt.legend(fontsize=30)

plt.xticks(fontsize=40)
plt.yticks(fontsize=40)



plt.tight_layout()

plt.savefig("rho_lambda01_tau_10_l0_03.pdf",dpi=1200)

fig1=plt.figure(figsize=(14,8))

plt.plot(r10012[0:][0],r10012[0:][1],color= "C0",label=r"$\lambda = 0.1,\tau=10,l_0=0.2$")

plt.plot(xx,np.sqrt(np.pi)*lorentzian_form(lamda2,tau2,xx)*ligando(x_bin,0.2),color="C1")

plt.xlabel(r'$x$',fontsize=30)

plt.ylabel(r'$\Delta\rho/\rho_0$',fontsize=30)

plt.legend(fontsize=30)

plt.xticks(fontsize=40)
plt.yticks(fontsize=40)



plt.tight_layout()

plt.savefig("rho_lambda01_tau_10_l0_02.pdf",dpi=1200)


fig1=plt.figure(figsize=(14,8))

plt.plot(r10011[0:][0],r10011[0:][1],color= "C0",label=r"$\lambda = 0.1,\tau=10,l_0=0.1$")

plt.plot(xx,np.sqrt(np.pi)*lorentzian_form(lamda2,tau2,xx)*ligando(x_bin,0.1),color="C1")

plt.xlabel(r'$x$',fontsize=30)

plt.ylabel(r'$\Delta\rho/\rho_0$',fontsize=30)

plt.legend(fontsize=30)

plt.xticks(fontsize=40)
plt.yticks(fontsize=40)



plt.tight_layout()

plt.savefig("rho_lambda01_tau_10_l0_01.pdf",dpi=1200)

fig1=plt.figure(figsize=(14,8))

plt.plot(r100110[0:][0],r100110[0:][1],color= "C0",label=r"$\lambda = 0.1,\tau=10,l_0=1.0$")

plt.plot(xx,np.sqrt(np.pi)*lorentzian_form(lamda2,tau2,xx)*ligando(x_bin,1.0),color="C1")

plt.xlabel(r'$x$',fontsize=30)

plt.ylabel(r'$\Delta\rho/\rho_0$',fontsize=30)

plt.legend(fontsize=30)

plt.xticks(fontsize=40)
plt.yticks(fontsize=40)



plt.tight_layout()

plt.savefig("rho_lambda01_tau_10_l0_1.pdf",dpi=1200)


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


# ????? FIGURAS para lambda=0.1 tau = 0.1


# r01015 =  elements[names.index('NormDensidadEscalon-0.1-0.1-0.5.dat')]  
# r01014 =  elements[names.index('NormDensidadEscalon-0.1-0.1-0.4.dat')] 
# r01013 =  elements[names.index('NormDensidadEscalon-0.1-0.1-0.3.dat')]  
# r01012 =  elements[names.index('NormDensidadEscalon-0.1-0.1-0.2.dat')]  
# r01011 =  elements[names.index('NormDensidadEscalon-0.1-0.1-0.1.dat')]  
# r010110 =  elements[names.index('NormDensidadEscalon-0.1-0.1-1.0.dat')]  




tau2 = 0.1
lamda2 =0.1


fig1=plt.figure(figsize=(14,8))

plt.plot(r01015[0:][0],r01015[0:][1],color= "C0",label=r"$\lambda = 0.1,\tau=0.1,l_0=0.5$")

plt.plot(xx,0.53*lorentzian_form(lamda2,tau2,xx)*ligando(x_bin,0.5),color="C1")

plt.xlabel(r'$x$',fontsize=30)

plt.ylabel(r'$\Delta\rho/\rho_0$',fontsize=30)

plt.legend(fontsize=30)

plt.xticks(fontsize=40)
plt.yticks(fontsize=40)



plt.tight_layout()

plt.savefig("rho_lambda01_tau_01_l0_05.pdf",dpi=1200)


fig1=plt.figure(figsize=(14,8))

plt.plot(r01014[0:][0],r01014[0:][1],color= "C0",label=r"$\lambda = 0.1,\tau=0.1,l_0=0.4$")

plt.plot(xx,0.53*lorentzian_form(lamda2,tau2,xx)*ligando(x_bin,0.4),color="C1")

plt.xlabel(r'$x$',fontsize=30)

plt.ylabel(r'$\Delta\rho/\rho_0$',fontsize=30)

plt.legend(fontsize=30)

plt.xticks(fontsize=40)
plt.yticks(fontsize=40)



plt.tight_layout()

plt.savefig("rho_lambda01_tau_01_l0_04.pdf",dpi=1200)


fig1=plt.figure(figsize=(14,8))

plt.plot(r01013[0:][0],r01013[0:][1],color= "C0",label=r"$\lambda = 0.1,\tau=0.1,l_0=0.3$")

plt.plot(xx,0.53*lorentzian_form(lamda2,tau2,xx)*ligando(x_bin,0.3),color="C1")

plt.xlabel(r'$x$',fontsize=30)

plt.ylabel(r'$\Delta\rho/\rho_0$',fontsize=30)

plt.legend(fontsize=30)

plt.xticks(fontsize=40)
plt.yticks(fontsize=40)



plt.tight_layout()

plt.savefig("rho_lambda01_tau_01_l0_03.pdf",dpi=1200)

fig1=plt.figure(figsize=(14,8))

plt.plot(r01012[0:][0],r01012[0:][1],color= "C0",label=r"$\lambda = 0.1,\tau=0.1,l_0=0.2$")

plt.plot(xx,0.53*lorentzian_form(lamda2,tau2,xx)*ligando(x_bin,0.2),color="C1")

plt.xlabel(r'$x$',fontsize=30)

plt.ylabel(r'$\Delta\rho/\rho_0$',fontsize=30)

plt.legend(fontsize=30)

plt.xticks(fontsize=40)
plt.yticks(fontsize=40)



plt.tight_layout()

plt.savefig("rho_lambda01_tau_01_l0_02.pdf",dpi=1200)

fig1=plt.figure(figsize=(14,8))

plt.plot(r01011[0:][0],r01011[0:][1],color= "C0",label=r"$\lambda = 0.1,\tau=0.1,l_0=0.1$")

plt.plot(xx,0.53*lorentzian_form(lamda2,tau2,xx)*ligando(x_bin,0.1),color="C1")

plt.xlabel(r'$x$',fontsize=30)

plt.ylabel(r'$\Delta\rho/\rho_0$',fontsize=30)

plt.legend(fontsize=30)

plt.xticks(fontsize=40)
plt.yticks(fontsize=40)



plt.tight_layout()

plt.savefig("rho_lambda01_tau_01_l0_01.pdf",dpi=1200)



fig1=plt.figure(figsize=(14,8))

plt.plot(r010110[0:][0],r010110[0:][1],color= "C0",label=r"$\lambda = 0.1,\tau=0.1,l_0=1.0$")

plt.plot(xx,0.53*lorentzian_form(lamda2,tau2,xx)*ligando(x_bin,1.0),color="C1")

plt.xlabel(r'$x$',fontsize=30)

plt.ylabel(r'$\Delta\rho/\rho_0$',fontsize=30)

plt.legend(fontsize=30)

plt.xticks(fontsize=40)
plt.yticks(fontsize=40)



plt.tight_layout()

plt.savefig("rho_lambda01_tau_01_l0_1.pdf",dpi=1200)

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!







fig1=plt.figure(figsize=(14,8))


# plt.plot(xx,ro1['dens'],color= "C0",label=r"$\lambda = 1$")

plt.plot(r0104[0:][0],r0104[0:][1],color= "C0",label=r"$\lambda = 1$")



plt.plot(xx,0.53*lorentzian_form(1,Tm,xx)*ligando(x_bin,l0),color="C1")
# plt.plot(xx,0.49*lorentzian_form_order_2(1,Tm,xx)*ligando(x_bin,l0),color="C1",linestyle="--")

# plt.plot(xx,0.082*ligando(x_bin,l0)/0.4,color="C1",linestyle="--")
# plt.plot(xx,0.5*lorentzian_form_articulo(1,Tm,xx)*ligando(x_bin,l0),color="C1",linestyle="--")
plt.plot(xxx,lorentz(xxx,popt1004[0],popt1004[1]),color="C1",linestyle="dashdot")


plt.xlabel(r'$x$',fontsize=30)

plt.ylabel(r'$\Delta\rho/\rho_0$',fontsize=30)

plt.legend(fontsize=30)

plt.xticks(fontsize=40)
plt.yticks(fontsize=40)



plt.tight_layout()

plt.savefig("rho_lambda1.pdf",dpi=1200)
fig1=plt.figure(figsize=(14,8))


# plt.plot(xx,ro1['dens'],color= "C0",label=r"$\lambda = 1$")

plt.plot(r0104[0:][0],r0104[0:][1],color= "C0",label=r"$\lambda = 1$")



plt.plot(xx,0.53*lorentzian_form(1,Tm,xx)*ligando(x_bin,l0),color="C1")
# plt.plot(xx,0.49*lorentzian_form_order_2(1,Tm,xx)*ligando(x_bin,l0),color="C1",linestyle="--")

# plt.plot(xx,0.082*ligando(x_bin,l0)/0.4,color="C1",linestyle="--")
# plt.plot(xx,0.5*lorentzian_form_articulo(1,Tm,xx)*ligando(x_bin,l0),color="C1",linestyle="--")
plt.plot(xxx,lorentz(xxx,popt1004[0],popt1004[1]),color="C1",linestyle="dashdot")


plt.xlabel(r'$x$',fontsize=30)

plt.ylabel(r'$\Delta\rho/\rho_0$',fontsize=30)

plt.legend(fontsize=30)

plt.xticks(fontsize=40)
plt.yticks(fontsize=40)



plt.tight_layout()

plt.savefig("rho_lambda1.pdf",dpi=1200)
fig1=plt.figure(figsize=(14,8))


# plt.plot(xx,ro1['dens'],color= "C0",label=r"$\lambda = 1$")

plt.plot(r0104[0:][0],r0104[0:][1],color= "C0",label=r"$\lambda = 1$")



plt.plot(xx,0.53*lorentzian_form(1,Tm,xx)*ligando(x_bin,l0),color="C1")
# plt.plot(xx,0.49*lorentzian_form_order_2(1,Tm,xx)*ligando(x_bin,l0),color="C1",linestyle="--")

# plt.plot(xx,0.082*ligando(x_bin,l0)/0.4,color="C1",linestyle="--")
# plt.plot(xx,0.5*lorentzian_form_articulo(1,Tm,xx)*ligando(x_bin,l0),color="C1",linestyle="--")
# plt.plot(xxx,lorentz(xxx,popt1004[0],popt1004[1]),color="C1",linestyle="dashdot")


plt.xlabel(r'$x$',fontsize=30)

plt.ylabel(r'$\Delta\rho/\rho_0$',fontsize=30)

plt.legend(fontsize=30)

plt.xticks(fontsize=40)
plt.yticks(fontsize=40)



plt.tight_layout()

plt.savefig("rho_lambda1.pdf",dpi=1200)
      
        
# ????? FIGURAS para lambda= 1



# r015
# r014
# r013
# r012
# r011
# r0110

fig1=plt.figure(figsize=(14,8))


# plt.plot(xx,ro1['dens'],color= "C0",label=r"$\lambda = 1$")

plt.plot(r015[0:][0],r015[0:][1],color= "C0",label=r"$\lambda = 1$")



plt.plot(xx,0.53*lorentzian_form(1,Tm,xx)*ligando(x_bin,l0),color="C1")
# plt.plot(xx,0.49*lorentzian_form_order_2(1,Tm,xx)*ligando(x_bin,l0),color="C1",linestyle="--")

# plt.plot(xx,0.082*ligando(x_bin,l0)/0.4,color="C1",linestyle="--")
# plt.plot(xx,0.5*lorentzian_form_articulo(1,Tm,xx)*ligando(x_bin,l0),color="C1",linestyle="--")
# plt.plot(xxx,lorentz(xxx,popt1004[0],popt1004[1]),color="C1",linestyle="dashdot")


plt.xlabel(r'$x$',fontsize=30)

plt.ylabel(r'$\Delta\rho/\rho_0$',fontsize=30)

plt.legend(fontsize=30)

plt.xticks(fontsize=40)
plt.yticks(fontsize=40)



plt.tight_layout()

plt.savefig("rho_lambda1.pdf",dpi=1200)

# ????? FIGURAS para lambda= 1 , tau = 0.1


fig1=plt.figure(figsize=(14,8))


# plt.plot(xx,ro1['dens'],color= "C0",label=r"$\lambda = 1$")

plt.plot(r010104[0:][0],r010104[0:][1],color= "C0",label=r"$\lambda = 1$")



plt.plot(xx,np.sqrt(1/np.pi)*lorentzian_form(1,0.1,xx)*ligando(x_bin,l0),color="C1")
# plt.plot(xx,0.49*lorentzian_form_order_2(1,Tm,xx)*ligando(x_bin,l0),color="C1",linestyle="--")

# plt.plot(xx,0.082*ligando(x_bin,l0)/0.4,color="C1",linestyle="--")
# plt.plot(xx,0.5*lorentzian_form_articulo(1,Tm,xx)*ligando(x_bin,l0),color="C1",linestyle="--")
# plt.plot(xxx,lorentz(xxx,k00101_media_2[2],k00101_media_2[0]),color="C1",linestyle="dashdot")


plt.xlabel(r'$x$',fontsize=30)

plt.ylabel(r'$\Delta\rho/\rho_0$',fontsize=30)

plt.legend(fontsize=30)

plt.xticks(fontsize=40)
plt.yticks(fontsize=40)



plt.tight_layout()

plt.savefig("rho_lambda1_tau=01.pdf",dpi=1200)






# ????? FIGURAS para lambda= 0.1

fig3=plt.figure(figsize=(14,8))


plt.plot(r014[0:][0],r014[0:][1],color= "C2",label=r"$\lambda = 0.1$")



plt.plot(xx,0.53*lorentzian_form(0.1,Tm,xx)*ligando(x_bin,l0),color="C3")

# plt.plot(xx,0.5*lorentzian_form_order_2(0.1,Tm,xx)*ligando(x_bin,l0),color="C3",linestyle="--")



# plt.plot(xx,max(r014[0:][1])*ligando(x_bin,l0)/0.4,color="C3",linestyle="--")

# plt.plot(xx,0.5*lorentzian_form_articulo(0.1,Tm,xx)*ligando(x_bin,l0),color="C3",linestyle="--")
plt.plot(xxx,lorentz(xxx,popt01[0],popt01[1]),color="C3",linestyle="--")



plt.xlabel(r'$x$',fontsize=30)

plt.ylabel(r'$\Delta\rho/\rho_0$',fontsize=30)

plt.legend(fontsize=30)

plt.xticks(fontsize=40)
plt.yticks(fontsize=40)

plt.tight_layout()


plt.savefig("rho_lambda01.pdf",dpi=1200)



# ????? FIGURAS para lambda= 0.2

fig102=plt.figure(figsize=(14,8))


# plt.plot(xx,ro1['dens'],color= "C0",label=r"$\lambda = 1$")

plt.plot(r02[0:][0],r02[0:][1],color= "C6",label=r"$\lambda = 0.2$")



plt.plot(xx,0.5*lorentzian_form(0.2,Tm,xx)*ligando(x_bin,l0),color="C7")
# plt.plot(xx,0.082*ligando(x_bin,l0)/0.4,color="C1",linestyle="--")
# plt.plot(xx,0.5*lorentzian_form_articulo(1,Tm,xx)*ligando(x_bin,l0),color="C1",linestyle="--")
plt.plot(xxx,lorentz(xxx,popt02[0],popt02[1]),color="C7",linestyle="--")


plt.xlabel(r'$x$',fontsize=30)

plt.ylabel(r'$\Delta\rho/\rho_0$',fontsize=30)

plt.legend(fontsize=40)

plt.xticks(fontsize=40)
plt.yticks(fontsize=40)


# ????? FIGURAS para lambda= 0.4

fig104=plt.figure(figsize=(14,8))


# plt.plot(xx,ro1['dens'],color= "C0",label=r"$\lambda = 1$")

plt.plot(r044[0:][0],r044[0:][1],color= "k",label=r"$\lambda = 0.4$")



plt.plot(xx,0.5*lorentzian_form(0.4,Tm,xx)*ligando(x_bin,l0),color="r")
# plt.plot(xx,0.082*ligando(x_bin,l0)/0.4,color="C1",linestyle="--")
# plt.plot(xx,0.5*lorentzian_form_articulo(1,Tm,xx)*ligando(x_bin,l0),color="C1",linestyle="--")
plt.plot(xxx,lorentz(xxx,popt04[0],popt04[1]),color="r",linestyle="--")


plt.xlabel(r'$x$',fontsize=30)

plt.ylabel(r'$\Delta\rho/\rho_0$',fontsize=30)

plt.legend(fontsize=40)

plt.xticks(fontsize=40)
plt.yticks(fontsize=40)



plt.tight_layout()

plt.savefig("rho_lambda04.pdf",dpi=1200)

  
# ????? FIGURAS para lambda= 0.5
 
fig34=plt.figure(figsize=(14,8))


plt.plot(r054[0:][0],r054[0:][1],color= "C4",label=r"$\lambda = 0.5$")



plt.plot(xx,0.5*lorentzian_form(0.5,Tm,xx)*ligando(x_bin,l0),color="C5")
plt.plot(xx,max(r054[0:][1])*ligando(x_bin,l0)/0.4,color="C5",linestyle="--")

# plt.plot(xx,0.5*lorentzian_form_articulo(0.5,Tm,xx)*ligando(x_bin,l0),color="C5",linestyle="--")
plt.plot(xxx,lorentz(xxx,popt05[0],popt05[1]),color="C5",linestyle="--")


plt.xlabel(r'$x$',fontsize=30)

plt.ylabel(r'$\Delta\rho/\rho_0$',fontsize=30)

plt.legend(fontsize=40)

plt.xticks(fontsize=40)
plt.yticks(fontsize=40)

plt.tight_layout()



plt.savefig("rho_lambda05.pdf",dpi=1200)


# ????? FIGURAS para varios lambda

# fig1234=plt.figure(figsize=(14,8))


# plt.plot(r014[0:][0],r014[0:][1],color= "C0",label=r"$\lambda = 0.5$")
# plt.plot(r014[0:][0],r014[0:][1],color= "C2",label=r"$\lambda = 0.5$")
# # plt.plot(r02[0:][0],r02[0:][1],color= "C4",label=r"$\lambda = 0.5$")
# # plt.plot(r044[0:][0],r044[0:][1],color= "C4",label=r"$\lambda = 0.5$")
# plt.plot(r054[0:][0],r054[0:][1],color= "C4",label=r"$\lambda = 0.5$")
# plt.plot(r1204[0:][0],r1204[0:][1],color= "C6",label=r"$\lambda = 0.5$")
# # plt.plot(r1504[0:][0],r1504[0:][1],color= "C10",label=r"$\lambda = 0.5$")




# plt.plot(xx,0.53*lorentzian_form(1,Tm,xx)*ligando(x_bin,l0),color="C1")
# plt.plot(xx,0.53*lorentzian_form(0.1,Tm,xx)*ligando(x_bin,l0),color="C3")

# # plt.plot(xx,0.53*lorentzian_form(0.2,Tm,xx)*ligando(x_bin,l0),color="C5")
# # plt.plot(xx,0.53*lorentzian_form(0.4,Tm,xx)*ligando(x_bin,l0),color="C7")


# plt.plot(xx,0.53*lorentzian_form(0.5,Tm,xx)*ligando(x_bin,l0),color="C5")
# # plt.plot(xx,0.53*lorentzian_form(1.0,Tm,xx)*ligando(x_bin,l0),color="C11")


# plt.plot(xx,0.53*lorentzian_form(1.2,Tm,xx)*ligando(x_bin,l0),color="C7")
# # plt.plot(xx,0.53*lorentzian_form(1.5,Tm,xx)*ligando(x_bin,l0),color="C13")

# # plt.plot(xx,max(r054[0:][1])*ligando(x_bin,l0)/0.4,color="C5",linestyle="--")

# # plt.plot(xx,0.5*lorentzian_form_articulo(0.5,Tm,xx)*ligando(x_bin,l0),color="C5",linestyle="--")
# # plt.plot(xxx,lorentz(xxx,popt05[0],popt05[1]),color="C5",linestyle="--")


# plt.xlabel(r'$x$',fontsize=30)

# plt.ylabel(r'$\Delta\rho/\rho_0$',fontsize=30)

# # plt.legend(fontsize=40)

# plt.xticks(fontsize=40)
# plt.yticks(fontsize=40)

# plt.tight_layout()



# plt.savefig("rho_lambda_varios.pdf",dpi=1200)



# ????? FIGURAS teoricas


fig3=plt.figure(figsize=(14,8))






plt.plot(xx,lorentzian_form(1,Tm,xx)*ligando(x_bin,l0),color="C0",label=r"$\lambda=1$")

plt.plot(xx,lorentzian_form_articulo(1,Tm,xx)*ligando(x_bin,l0),color="C0",linestyle="--",label=r"$\lambda=1$")

plt.plot(xx,lorentzian_form(0.5,Tm,xx)*ligando(x_bin,l0),color="C1",label=r"$\lambda=0.5$")

plt.plot(xx,lorentzian_form_articulo(0.5,Tm,xx)*ligando(x_bin,l0),color="C1",linestyle="--",label=r"$\lambda=0.5$")


plt.plot(xx,lorentzian_form(0.1,Tm,xx)*ligando(x_bin,l0),color="C2",label=r"$\lambda=0.1$")

plt.plot(xx,lorentzian_form_articulo(0.1,Tm,xx)*ligando(x_bin,l0),color="C2",linestyle="--",label=r"$\lambda=0.1$")


plt.xlabel(r'$x$',fontsize=40)

plt.ylabel(r'$\rho(x)-\rho_0$',fontsize=40)

plt.legend()

plt.xticks(fontsize=25)
plt.yticks(fontsize=25)



plt.tight_layout()



fig3=plt.figure(figsize=(14,8))


lmbs=np.linspace(0,2.1,100)

lmb_array= [0.1,0.2,0.4,0.5,0.7,1.0,1.2,1.5,2.0]




long_array=[1/popt01[1],1/popt02[1],1/popt04[1],1/popt05[1],1/popt07[1],1/popt1004[1],1/popt12[1],1/popt15[1]]
long_error_array=[(1/popt01[1])**2*np.diag(pcov01)[1],(1/popt02[1])**2*np.diag(pcov02)[1],(1/popt04[1])**2*np.diag(pcov04)[1],(1/popt05[1])**2*np.diag(pcov05)[1],(1/popt07[1])**2*np.diag(pcov07)[1],(1/popt1004[1])**2*np.diag(pcov1004)[1],(1/popt12[1])**2*np.diag(pcov12)[1],(1/popt15[1])**2*np.diag(pcov15)[1]]


k0_array=[popt01[1],popt02[1],popt04[1],popt05[1],popt07[1],popt1004[1],popt12[1],popt15[1],popt2[1]]
k0_teoricos = [k0(0.1,Tm),k0(0.2,Tm),k0(0.4,Tm),k0(0.5,Tm),k0(0.7,Tm),k0(1.0,Tm),k0(1.2,Tm),k0(1.5,Tm),k0(2.0,Tm)]

k0_error_array=[np.diag(pcov01)[1],np.diag(pcov02)[1],np.diag(pcov04)[1],np.diag(pcov05)[1],np.diag(pcov07)[1],np.diag(pcov1004)[1],np.diag(pcov12)[1],np.diag(pcov15)[1],np.diag(pcov2)[1]]



print("Array de los k0 ",k0_array)
print("Array de los k0 teoricos ",k0_teoricos)
k0_array03=[popt103[1],popt203[1],popt403[1],popt503[1],popt703[1],popt1003[1],popt1203[1],popt1503[1],popt2003[1]]
k0_array02=[popt102[1],popt202[1],popt402[1],popt502[1],popt702[1],popt1002[1],popt1202[1],popt1502[1],popt2002[1]]
k0_array04=[popt104[1],popt204[1],popt404[1],popt504[1],popt704[1],popt1004[1],popt1204[1],popt1504[1],popt2004[1]]
k0_array01=[popt101[1],popt201[1],popt401[1],popt501[1],popt701[1],popt1001[1],popt1201[1],popt1501[1],popt2001[1]]
k0_array05=[popt105[1],popt205[1],popt405[1],popt505[1],popt705[1],popt1005[1],popt1205[1],popt1505[1],popt2005[1]]
k0_array10=[popt1010[1],popt2010[1],popt4010[1],popt5010[1],popt7010[1],popt10010[1],popt12010[1],popt15010[1],popt20010[1]]



k0_error_array2=[np.diag(pcov105)[1],np.diag(pcov103)[1],np.diag(pcov102)[1],np.diag(pcov101)[1],np.diag(pcov1004)[1]]
# lmb_array2 = np.ones(len(k0_array2))
# print("suma errores",sum(k0_error_array2)/5)
# k0_array3=[popt023[1],popt025[1],popt0210[1]]
# k0_error_array3=[np.diag(pcov023)[1],np.diag(pcov025)[1],np.diag(pcov0210)[1]]
# lmb_array3 = np.ones(len(k0_array3))*0.2
print(np.diag(pcov103)+np.diag(pcov105)+np.diag(pcov102)+np.diag(pcov101)+np.diag(pcov1004))
print(k0_error_array2)

k0_medias = [k01_media,k002_media,k04_media,k05_media,k07_media,k010_media,k012_media,k015_media,k02_media]
k0_error_medias =[sd_k01,sd_k002,sd_k04,sd_k05,sd_k07,sd_k010,sd_k012,sd_k015,sd_k02]


k0_medias_2 = [k001_media_2[0],k002_media_2[0],k004_media_2[0],k005_media_2[0],k007_media_2[0],k010_media_2[0],k0012_media_2[0],k0015_media_2[0],k0020_media_2[0]]
k0_error_medias_2=[k001_media_2[1],k002_media_2[1],k004_media_2[1],k005_media_2[1],k007_media_2[1],k010_media_2[1],k0012_media_2[1],k0015_media_2[1],k0020_media_2[1]]

k0_medias_2_01 = [k01010_media_2[0],
k01002_media_2[0],
k01001_media_2[0],
k01004_media_2[0],
k01005_media_2[0],
k01007_media_2[0],
k010012_media_2[0],
k010015_media_2[0],
k010020_media_2[0]]
k0_error_medias_2_01 = [k01010_media_2[1],
k01002_media_2[1],
k01001_media_2[1],
k01004_media_2[1],
k01005_media_2[1],
k01007_media_2[1],
k010012_media_2[1],
k010015_media_2[1],
k010020_media_2[1]]

k0_medias_2_1=[k00110_media_2[0],
k00102_media_2[0],
k00104_media_2[0],
k00105_media_2[0],
k00107_media_2[0],
k00101_media_2[0],
k001012_media_2[0],
k001015_media_2[0],
k001020_media_2[0]]
k0_error_medias_2_1=[k00110_media_2[1],
k00102_media_2[1],
k00104_media_2[1],
k00105_media_2[1],
k00107_media_2[1],
k00101_media_2[1],
k001012_media_2[1],
k001015_media_2[1],
k001020_media_2[1]]

print("Array de los k0 ",k0_medias)
print("Array de los k0 teoricos ",k0_teoricos)

# ??? TAU = 1 

fig3=plt.figure(figsize=(14,8))

plt.plot(lmbs,k0(lmbs,Tm),color="k")

plt.errorbar(lmb_array,k0_medias_2, yerr=k0_error_medias_2,mfc="none",capsize=10,ms=12, color='b',marker="^",linestyle="",label= r" $\tau_{sim} =  1.0 $ "  ) 

plt.xlabel(r'$\lambda$',fontsize=30)

plt.ylabel(r'$k_0$',fontsize=30)

plt.legend(fontsize=30,loc="best")

plt.xticks(fontsize=40)
plt.yticks(fontsize=40)



plt.xlim(0.0,2.1)


plt.tight_layout()
plt.savefig("k0_lambda_tau1.pdf",dpi=1200)



plt.show()




# ??? TAU = 10.0 

fig3=plt.figure(figsize=(14,8))

plt.plot(lmbs,k0(lmbs,10.0),color="k")

plt.errorbar(lmb_array,k0_medias_2_01, yerr=k0_error_medias_2_01,mfc="none",capsize=10,ms=12, color='g',marker="o",linestyle="",label= r" $\tau_{sim} =  1.0 $ "  ) 



plt.xlabel(r'$\lambda$',fontsize=30)

plt.ylabel(r'$k_0$',fontsize=30)

plt.legend(fontsize=30,loc="best")

plt.xticks(fontsize=40)
plt.yticks(fontsize=40)



plt.xlim(0.0,2.1)


plt.tight_layout()
plt.savefig("k0_lambda_tau10.pdf",dpi=1200)



plt.show()



# ??? TAU = 0.1 

fig3=plt.figure(figsize=(14,8))

plt.plot(lmbs,k0(lmbs,0.1),color="k")

plt.errorbar(lmb_array,k0_medias_2_1, yerr=k0_error_medias_2_1,mfc="none",capsize=10,ms=12, color='r',marker="s",linestyle="",label= r" $\tau_{sim} =  0.1$ "    ) 



plt.xlabel(r'$\lambda$',fontsize=30)

plt.ylabel(r'$k_0$',fontsize=30)

plt.legend(fontsize=30,loc="best")

plt.xticks(fontsize=40)
plt.yticks(fontsize=40)



plt.xlim(0.0,2.1)


plt.tight_layout()
plt.savefig("k0_lambda_tau01.pdf",dpi=1200)



plt.show()

fig3=plt.figure(figsize=(14,8))

plt.plot(lmbs,k0(lmbs,Tm)/(np.sqrt(1+Tm)/Tm),color="k")


plt.plot(lmbs,k0(lmbs,0.1)/(np.sqrt(1+0.1)/0.1),color="k",linestyle="dashdot")

plt.plot(lmbs,k0(lmbs,10.0)/(np.sqrt(1+10.0)/10.0),color="k",linestyle="dotted")

plt.errorbar(lmb_array,k0_medias_2/(np.sqrt(1+Tm)/Tm), yerr=k0_error_medias_2,mfc="none",capsize=10,ms=12, color='b',marker="^",linestyle="",label= r" $\tau_{sim} =  1.0 $ "  ) 

plt.errorbar(lmb_array,k0_medias_2_01/(np.sqrt(1+10.0)/10.0), yerr=k0_error_medias_2_01,mfc="none",capsize=10,ms=12, color='g',marker="o",linestyle="",label= r" $\tau_{sim} =  10.0 $ "   ) 

plt.errorbar(lmb_array,k0_medias_2_1/(np.sqrt(1+0.1)/0.1), yerr=k0_error_medias_2_1,mfc="none",capsize=10,ms=12, color='r',marker="s",linestyle="",label= r" $\tau_{sim} =  0.1$ "    ) 



plt.xlabel(r'$\lambda$',fontsize=30)

plt.ylabel(r'$k_0/k^*_0$',fontsize=30)

plt.legend(fontsize=30,loc="best")

plt.xticks(fontsize=40)
plt.yticks(fontsize=40)



plt.xlim(0.0,2.1)
# plt.ylim(0,k0(2.0,Tm))


plt.tight_layout()
plt.savefig("k0_lambda_varios.pdf",dpi=1200)



plt.show()