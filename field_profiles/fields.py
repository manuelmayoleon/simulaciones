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
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, InsetPosition,
                                                  mark_inset)
# import scipy.special as special

# Import math Library
import math 
import numpy.ma as ma
from sympy import *

import csv
import os 
import glob

from matplotlib.ticker import FormatStrFormatter

from scipy.optimize import curve_fit

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


path = os.getcwd()
path=path + "/field_profiles/"
print(path)
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
print(elements[0]) 


def buscar_arrays(word_to_check):
    
    matching_arrays = [array_name for array_name in names if word_to_check in array_name]
    arrays = list()
    if matching_arrays:
        print(f"The word '{word_to_check}' is present in the following array names:")
        matching_arrays = sorted(matching_arrays)
        for array_name in matching_arrays:
            arrays.append(elements[names.index(array_name)]) 
            print(array_name)
    else:
         print(f"The word '{word_to_check}' is not present in any of the array names.")
    
    return arrays





# !! PARA tau =0.1 la longitud de la caja es L=10. Tenemos comida l=0.1,0.2,0.3,0.4,0.5,1.0


# !! rho para tau = 0.1 lambda=0.1
tau01_lamda01 = list()
word_to_check = "DensidadEscalon-0.1-0.1"

tau01_lamda01 = buscar_arrays(word_to_check)
# print(tau01_lamda01[0])

# !! rho para tau = 0.1 lambda=0.2
tau01_lamda02 = list()
word_to_check = "DensidadEscalon-0.1-0.2"

tau01_lamda02 = buscar_arrays(word_to_check)
# print(tau01_lamda02[0])







# ?? PARA tau =1.0 la longitud de la caja es L=20. Tenemos comida l=0.1,0.2,0.3,0.4,0.5,1.0


# !! rho para tau = 1.0 lambda=0.1
tau1_lamda1 = list()
word_to_check = "DensidadEscalon-1.0-0.1"

tau1_lamda1 = elements[0]
print(tau1_lamda1[0])







N = 1000 # 2000

# Parametros del sistema

v0 = V = B = D =  1
lamda = 1.0
l0=0.4

############## Valores gradiente y tiempo de memoria

Tm = 1.0

##############

L = 20.0

dx = L/10000

# Arreglo espacial de la configuracion

xfield = np.arange(0,L,dx)

x_bin = xfield + .5*dx


xx = np.arange(-L/2,L/2,dx)


# !!!!!!!!!!!! FUNCIONES !!!
def ligando_escalar(x,l0,L):
    
#    return math.tanh(4*math.sin(2*math.pi*x/L))

    if(x<L/2):

        return -l0

    else:

        return l0


def lorentzian_form(lamb,tau,x,L):
    
    psi0=lamb*np.exp(lamb**2/2)/(1+np.exp(lamb**2/2)*tau*(1+lamb**2))
    k0 = np.sqrt((1+np.exp(lamb**2/2)*tau*(1+lamb**2))/tau**2)
    if(x>L/4 or x<(-L/4)):
        return  psi0*(1 - np.exp(- k0* abs(x-np.sign(x)*L*0.5))) 
    else:
        return  psi0*(1 - np.exp(- k0* abs(x)))

def mu0_2(lamb,tau): 
    return (lamb * (2 + np.exp(lamb**2/2) * (1 + lamb**2) *  tau))/(2 + np.exp(lamb**2/2) *  (3 + 2  * lamb**2) *  tau + np.exp(lamb**2) * tau**2)
def mup0_2(lamb,tau): 
    return ( (2 + np.exp(lamb**2/2) * (1 + 2*lamb**2) *  tau))/(2 + np.exp(lamb**2/2) *  (3 + 2  * lamb**2) *  tau + np.exp(lamb**2) * tau**2)
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
  
   return  psi0(lamb,tau)*(1 - np.exp(- k0_2(lamb,tau)* abs(x)))/ k0_2(lamb,tau)

# !! Definicion de la función respuesta de rho_X 

# def A(lamb,tau):
    
#    return 0.5*Dx_2(lamb,tau)*np.sqrt(np.pi*0.5)/(D_2(lamb,tau)*DDx_2(lamb,tau)-2*Dx_2(lamb,tau))

# def C(lamb,tau):
    
#    return mu0_2(lamb,tau)*DDx_2(lamb,tau)**2 + mup0_2(lamb,tau)*(DDx_2(lamb,tau)**2-2*Dx_2(lamb,tau)**2)


def resp_rhox_nop(lamb,tau,x,L):
    return - np.exp(-abs(x)*k0_2(lamb,tau))*np.sqrt(np.pi*0.5)*mu0_2(lamb,tau)/Dx_2(lamb,tau)


def resp_rhox(lamb,tau,x,L):
    
    if(x>L/4 or x<(-L/4)):

        return  - np.exp(- k0_2(lamb,tau)* abs(x-np.sign(x)*L*0.5)) *np.sqrt(np.pi*0.5)*mu0_2(lamb,tau)/Dx_2(lamb,tau)
    
    else:

        return  - np.exp(- k0_2(lamb,tau)* abs(x)) *np.sqrt(np.pi*0.5)*mu0_2(lamb,tau)/Dx_2(lamb,tau)


def resp_x(lamb,tau,x,l1,L):
    
    if(x>L/4 or x<(-L/4)):

        return  (- l1 * np.sign(x) * np.exp(- k0_2(lamb,tau)* abs(x-np.sign(x)*L*0.5)) *np.sqrt(np.pi*0.5)*mu0_2(lamb,tau)/Dx_2(lamb,tau)) / ( 1+ 0.5 * l1 * np.sign(x) * psi0(lamb,tau)*(1 - np.exp( - k0_2(lamb,tau)* abs(x) ) ) )
    
    else:

        return  (- l1 * np.sign(x) * np.exp( - k0_2(lamb,tau)* abs(x) ) *np.sqrt(np.pi*0.5)*mu0_2(lamb,tau)/Dx_2(lamb,tau)) / ( 1+ 0.5*  l1 * np.sign(x) * psi0(lamb,tau)*(1 - np.exp(- k0_2(lamb,tau)*abs(x))))



def lorentzian_form_articulo(lamb,tau,x):
    
   return lamb*tau*(1 - np.exp(-np.sqrt(1+tau)*abs(x)/tau))/(1+tau)


ligando = np.vectorize(ligando_escalar)

lorentz_fr = np.vectorize(lorentzian_form)

lorentz_rhox = np.vectorize(resp_rhox)

xmean = np.vectorize(resp_x) 

def lorentz(x,a,b):
   return  a*np.sign(x)*(1 - np.exp(- b* abs(x)))


def exponential(x,a,b):
   return  a* np.exp(- b* x)



def ro_prime(vector):
    
    i=0
    ro_prima=np.zeros(int(len(xx)/4))
    for x in (vector['pos']):
        if x<L/4:
        
            ro_prima[i] = 0.5*( vector['dens'][list(vector['pos']).index(x)] + vector['dens'][list(vector['pos']).index(find_nearest(vector['pos'], L/2-x))] - vector['dens'][list(vector['pos']).index(find_nearest(vector['pos'], x+3*L/2))] + vector['dens'][list(vector['pos']).index(find_nearest(vector['pos'],L- x))]  )

        i+=1
       
    return ro_prima
def ro_pp(vector,L):
    
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





dx = L/10000

xfield = np.arange(0,L,dx)

x_bin = xfield + .5*dx


xx = np.arange(-L/2,L/2,dx)


fig1=plt.figure(figsize=(14,8))


plt.plot(tau1_lamda1[0],tau1_lamda1[2]/7.8,color= "C0",marker="o",linestyle="",markersize=5.0)
plt.plot(xx,lorentz_rhox(lamda,Tm,xx,L)*ligando(x_bin,0.4,L),color="r",linestyle="-",linewidth=2.5)



plt.xlabel(r'$x$',fontsize=40)

plt.ylabel(r'$\rho_X/\rho_0$',fontsize=40)

plt.xlim(-10.0,10.0)

plt.xticks(fontsize=25)
plt.yticks(fontsize=25)



plt.tight_layout()


#plt.savefig('response_to_step_function_tau'+str(Tm)+'lambda_'+str(lamda)+'.pdf',dpi=1200)
plt.savefig('rhox_response_to_step_function_tau'+str(Tm)+'lambda_'+str(lamda)+'.pdf',dpi=1200)


fig11=plt.figure(figsize=(14,8))

plt.plot(tau1_lamda1[0],tau1_lamda1[1],color= "C0",marker="o",linestyle="",markersize=5.0)
plt.plot(xx,0.53*lorentz_fr(lamda,Tm,xx,L)*ligando(x_bin,0.4,L),color="r",linestyle="-",linewidth=2.5)


plt.xlabel(r'$x$',fontsize=40)

plt.ylabel(r'$\Delta\rho/\rho_0$',fontsize=40)

#plt.legend(fontsize=30)

plt.xticks(fontsize=25)
plt.yticks(fontsize=25)

plt.xlim(-10.0,10.0)

plt.tight_layout()


plt.savefig('response_to_step_function_tau'+str(Tm)+'lambda_'+str(lamda)+'.pdf',dpi=1200)



fig12=plt.figure(figsize=(14,8))

plt.plot(tau1_lamda1[0],tau1_lamda1[3],color= "C0",marker="o",linestyle="",markersize=5.0)
plt.plot(xx,xmean(lamda,Tm,xx,l0,L)*0.4,color="r",linestyle="-",linewidth=2.5)


plt.xlabel(r'$x$',fontsize=40)

plt.ylabel(r'$\langle X \rangle $',fontsize=40)

#plt.legend(fontsize=30)

plt.xticks(fontsize=25)
plt.yticks(fontsize=25)

plt.xlim(-10.0,10.0)

plt.tight_layout()


plt.savefig('xmean_response_to_step_function_tau'+str(Tm)+'lambda_'+str(lamda)+'.pdf',dpi=1200)


fig13=plt.figure(figsize=(14,8))

plt.plot(tau1_lamda1[0],tau1_lamda1[4],color= "C0",marker="o",linestyle="",markersize=5.0)
plt.plot(xx,np.exp(lamda**2*0.5 + lamda*xmean(lamda,Tm,xx,l0,L)*0.4),color="r",linestyle="-",linewidth=2.5)
plt.plot(tau1_lamda1[0],np.exp(lamda**2*0.5 + lamda*tau1_lamda1[3]),color="lime",linestyle="--",linewidth=2.5)


plt.xlabel(r'$x$',fontsize=40)

plt.ylabel(r'$\langle \nu \rangle $',fontsize=40)

#plt.legend(fontsize=30)

plt.xticks(fontsize=25)
plt.yticks(fontsize=25)

plt.xlim(-10.0,10.0)

plt.tight_layout()


plt.savefig('numean_response_to_step_function_tau'+str(Tm)+'lambda_'+str(lamda)+'.pdf',dpi=1200)



plt.show()