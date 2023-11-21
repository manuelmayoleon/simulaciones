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
path = path+"/ResultadosFinales/"
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
print(names.index('DensidadEscalon-0.1-0.1-0.1-10.0.dat'))

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




# !! rho para tau = 0.1 lambda=0.4
tau01_lamda04 = list()
word_to_check = "DensidadEscalon-0.1-0.4"

tau01_lamda04 = buscar_arrays(word_to_check)
# print(tau01_lamda04[0])

# !! rho para tau = 0.1 lambda=0.5
tau01_lamda05 = list()
word_to_check = "DensidadEscalon-0.1-0.5"

tau01_lamda05 = buscar_arrays(word_to_check)
# print(tau01_lamda04[0])


# !! rho para tau = 0.1 lambda=0.7
tau01_lamda07 = list()
word_to_check = "DensidadEscalon-0.1-0.7"

tau01_lamda07 = buscar_arrays(word_to_check)
# print(tau01_lamda07[0])


# !! rho para tau = 0.1 lambda=1.0
tau01_lamda10 = list()
word_to_check = "DensidadEscalon-0.1-1.0"

tau01_lamda10 = buscar_arrays(word_to_check)
# print(tau01_lamda10[0])



# !! rho para tau = 0.1 lambda=1.2
tau01_lamda12 = list()
word_to_check = "DensidadEscalon-0.1-1.2"

tau01_lamda12 = buscar_arrays(word_to_check)
# print(tau01_lamda12[0])


# !! rho para tau = 0.1 lambda=1.5
tau01_lamda15 = list()
word_to_check = "DensidadEscalon-0.1-1.5"

tau01_lamda15 = buscar_arrays(word_to_check)
# print(tau01_lamda15[0])

# !! rho para tau = 0.1 lambda=2.0
tau01_lamda20 = list()
word_to_check = "DensidadEscalon-0.1-2.0"

tau01_lamda20 = buscar_arrays(word_to_check)
# print(tau01_lamda20[0])

# ?? PARA tau =10.0 la longitud de la caja es L=50. Tenemos comida l=0.1,0.2,0.3,0.4,0.5,1.0


# !! rho para tau = 10.0 lambda=0.1
tau10_lamda01 = list()
word_to_check = "DensidadEscalon-10.0-0.1"

tau10_lamda01 = buscar_arrays(word_to_check)
# print(tau01_lamda01[0])

# !! rho para tau = 10.0 lambda=0.2
tau10_lamda02 = list()
word_to_check = "DensidadEscalon-10.0-0.2"

tau10_lamda02 = buscar_arrays(word_to_check)
# print(tau01_lamda02[0])






# !! rho para tau = 10.0 lambda=0.4
tau10_lamda04 = list()
word_to_check = "DensidadEscalon-10.0-0.4"

tau10_lamda04 = buscar_arrays(word_to_check)
# print(tau01_lamda04[0])

# !! rho para tau = 10.0 lambda=0.5
tau10_lamda05 = list()
word_to_check = "DensidadEscalon-10.0-0.5"

tau10_lamda05 = buscar_arrays(word_to_check)
# print(tau01_lamda04[0])


# !! rho para tau = 10.0 lambda=0.7
tau10_lamda07 = list()
word_to_check = "DensidadEscalon-10.0-0.7"

tau10_lamda07 = buscar_arrays(word_to_check)
# print(tau01_lamda07[0])


# !! rho para tau = 10.0 lambda=1.0
tau10_lamda10 = list()
word_to_check = "DensidadEscalon-10.0-1.0"

tau10_lamda10 = buscar_arrays(word_to_check)
print(tau10_lamda10[0])



# !! rho para tau = 10.0 lambda=1.2
tau10_lamda12 = list()
word_to_check = "DensidadEscalon-10.0-1.2"

tau10_lamda12 = buscar_arrays(word_to_check)
# print(tau01_lamda12[0])


# !! rho para tau = 10.0 lambda=1.5
tau10_lamda15 = list()
word_to_check = "DensidadEscalon-10.0-1.5"

tau10_lamda15 = buscar_arrays(word_to_check)
# print(tau01_lamda15[0])

# !! rho para tau = 10.0 lambda=2.0
tau10_lamda20 = list()
word_to_check = "DensidadEscalon-10.0-2.0"

tau10_lamda20 = buscar_arrays(word_to_check)
# print(tau01_lamda20[0])



# ?? PARA tau =1.0 la longitud de la caja es L=20. Tenemos comida l=0.1,0.2,0.3,0.4,0.5,1.0


# !! rho para tau = 10.0 lambda=0.1
tau1_lamda01 = list()
word_to_check = "DensidadEscalon-1.0-0.1"

tau1_lamda01 = buscar_arrays(word_to_check)
# print(tau01_lamda01[0])

# !! rho para tau = 1.0 lambda=0.2
tau1_lamda02 = list()
word_to_check = "DensidadEscalon-1.0-0.2"

tau1_lamda02 = buscar_arrays(word_to_check)
# print(tau01_lamda02[0])






# !! rho para tau = 1.0 lambda=0.4
tau1_lamda04 = list()
word_to_check = "DensidadEscalon-1.0-0.4"

tau1_lamda04 = buscar_arrays(word_to_check)
# print(tau01_lamda04[0])

# !! rho para tau = 1.0 lambda=0.5
tau1_lamda05 = list()
word_to_check = "DensidadEscalon-1.0-0.5"

tau1_lamda05 = buscar_arrays(word_to_check)
# print(tau01_lamda04[0])


# !! rho para tau = 1.0 lambda=0.7
tau1_lamda07 = list()
word_to_check = "DensidadEscalon-1.0-0.7"

tau1_lamda07 = buscar_arrays(word_to_check)
# print(tau01_lamda07[0])


# !! rho para tau = 1.0 lambda=1.0
tau1_lamda10 = list()
word_to_check = "DensidadEscalon-1.0-1.0"

tau1_lamda10 = buscar_arrays(word_to_check)
print(tau10_lamda10[0])



# !! rho para tau = 1.0 lambda=1.2
tau1_lamda12 = list()
word_to_check = "DensidadEscalon-1.0-1.2"

tau1_lamda12 = buscar_arrays(word_to_check)
# print(tau01_lamda12[0])


# !! rho para tau = 1.0 lambda=1.5
tau1_lamda15 = list()
word_to_check = "DensidadEscalon-1.0-1.5"

tau1_lamda15 = buscar_arrays(word_to_check)
# print(tau01_lamda15[0])

# !! rho para tau = 1.0 lambda=2.0
tau1_lamda20 = list()
word_to_check = "DensidadEscalon-1.0-2.0"

tau1_lamda20 = buscar_arrays(word_to_check)
# print(tau01_lamda20[0])




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





def lorentzian_form_articulo(lamb,tau,x):
    
   return lamb*tau*(1 - np.exp(-np.sqrt(1+tau)*abs(x)/tau))/(1+tau)


ligando = np.vectorize(ligando_escalar)
lorentz_fr = np.vectorize(lorentzian_form)


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



def calculo_k0_simulaciones(vector,L,index):
    
    tt_01_10 = obtener_k0_2(vector[5],L,index)
    tt_01_05 = obtener_k0_2(vector[4],L,index)
    tt_01_04 = obtener_k0_2(vector[3],L,index)
    tt_01_03 = obtener_k0_2(vector[2],L,index)
    tt_01_02 = obtener_k0_2(vector[1],L,index)
    tt_01_01 = obtener_k0_2(vector[0],L,index)


    popt5, pcov5 = curve_fit(lorentz,np.reshape(tt_01_05[1], tt_01_05[1].size), np.reshape(tt_01_05[0], tt_01_05[0].size))
    popt4, pcov4 = curve_fit(lorentz,np.reshape(tt_01_04[1], tt_01_04[1].size), np.reshape(tt_01_04[0], tt_01_04[0].size))
    popt3, pcov3 = curve_fit(lorentz,np.reshape(tt_01_03[1], tt_01_03[1].size), np.reshape(tt_01_03[0], tt_01_03[0].size))
    popt2, pcov2 = curve_fit(lorentz,np.reshape(tt_01_02[1], tt_01_02[1].size), np.reshape(tt_01_02[0], tt_01_02[0].size))
    popt1, pcov1 = curve_fit(lorentz,np.reshape(tt_01_01[1], tt_01_01[1].size), np.reshape(tt_01_01[0], tt_01_01[0].size))
    popt10, pcov10 = curve_fit(lorentz,np.reshape(tt_01_10[1], tt_01_10[1].size), np.reshape(tt_01_10[0], tt_01_10[0].size))

    k010_media = ( popt5[1]+popt4[1]+popt3[1]+popt2[1]+popt1[1]+popt10[1])/6 
    sd_k010 = np.sqrt((popt5[1]**2+popt3[1]**2+popt2[1]**2+popt1[1]**2+popt4[1]**2+popt10[1]**2)/6-k010_media**2)

    
    return [k010_media,sd_k010]



def calculo_k0_simulaciones_rhox(vector,L,index):
    
    tt_01_10 = obtener_k0(vector[5],L,index)
    tt_01_05 = obtener_k0(vector[4],L,index)
    tt_01_04 = obtener_k0(vector[3],L,index)
    tt_01_03 = obtener_k0(vector[2],L,index)
    tt_01_02 = obtener_k0(vector[1],L,index)
    tt_01_01 = obtener_k0(vector[0],L,index)


    popt5, pcov5 = curve_fit(exponential,np.reshape(tt_01_05[1], tt_01_05[1].size), np.reshape(-tt_01_05[0], tt_01_05[0].size))
    popt4, pcov4 = curve_fit(exponential,np.reshape(tt_01_04[1], tt_01_04[1].size), np.reshape(-tt_01_04[0], tt_01_04[0].size))
    popt3, pcov3 = curve_fit(exponential,np.reshape(tt_01_03[1], tt_01_03[1].size), np.reshape(-tt_01_03[0], tt_01_03[0].size))
    popt2, pcov2 = curve_fit(exponential,np.reshape(tt_01_02[1], tt_01_02[1].size), np.reshape(-tt_01_02[0], tt_01_02[0].size))
    popt1, pcov1 = curve_fit(exponential,np.reshape(tt_01_01[1], tt_01_01[1].size), np.reshape(-tt_01_01[0], tt_01_01[0].size))
    popt10, pcov10 = curve_fit(exponential,np.reshape(tt_01_10[1], tt_01_10[1].size), np.reshape(-tt_01_10[0], tt_01_10[0].size))

    k010_media = ( popt5[1]+popt4[1]+popt3[1]+popt2[1]+popt1[1]+popt10[1])/6 
    sd_k010 = np.sqrt((popt5[1]**2+popt3[1]**2+popt2[1]**2+popt1[1]**2+popt4[1]**2+popt10[1]**2)/6-k010_media**2)

    
    return [k010_media,sd_k010]


def obtener_k0(vector,L,index):
    # ! index = 1 is rho
    # ! index = 2s is rho_X
    i=0
    ro_pr=np.zeros([int(len(vector[0])/4),1])
    pos=np.zeros([int(len(vector[0])/4),1])
    # print(int(len(vector[2])/4))
    for x in (vector[0]):
        # print(x)
        if  x<L/4 and x>0:
            # print(i)
            ro_pr[i] =float(vector.loc[vector[0] == x][index])
            pos[i] =float(x)
            # print(ro_pr)
            i+=1
       
    return ro_pr,pos

def obtener_k0_2(vector,L,index):
    # ! index = 1 is rho
    # ! index = 2s is rho_X
    i=0
    ro_pr=np.zeros([int(len(vector[0])/2),1])
    pos=np.zeros([int(len(vector[0])/2),1])
    # print(int(len(vector[2])/4))
    for x in (vector[0]):
        # print(x)
        if  abs(x)<L/4:
            # print(i)
            ro_pr[i] =float(vector.loc[vector[0] == x][index])
            pos[i] =float(x)
            # print(ro_pr)
            i+=1
       
    return ro_pr,pos




# ?   ?????????????????????????????????????????  #

# ? Figuras para tau =10.0 l

#? Definimos L, y los vectores longitud

L=50
Tm = 10.0

#? Arreglo espacial de la configuracion 

dx = L/300

xfield = np.arange(0,L,dx)

x_bin = xfield + .5*dx


xx = np.arange(-L/2,L/2,dx)

# ? Calculo de k0 para tau =10.0 lambda = 0.1
# ? definimos el valor de lambda y del ligando

lamda =0.1
ligand = 0.5


[media_k0_tau10_lamda01, sd_k0_tau10_lamda01 ] = calculo_k0_simulaciones(tau10_lamda01,L,1)

print(media_k0_tau10_lamda01)
print(sd_k0_tau10_lamda01)
print(k0_2(lamda,Tm))



# ? Calculo de k0 para tau =0.1 lambda = 0.2
# ? definimos el valor de lambda y del ligando

lamda =0.2
ligand = 0.5


[media_k0_tau10_lamda02, sd_k0_tau10_lamda02 ] = calculo_k0_simulaciones(tau10_lamda02,L,1)

print(media_k0_tau10_lamda02)
print(sd_k0_tau10_lamda02)
print(k0_2(lamda,Tm))


# ? Calculo de k0 para tau =0.1 lambda = 0.4
# ? definimos el valor de lambda y del ligando

lamda =0.4
ligand = 0.5


[media_k0_tau10_lamda04, sd_k0_tau10_lamda04 ] = calculo_k0_simulaciones(tau10_lamda04,L,1)

print(media_k0_tau10_lamda04)
print(sd_k0_tau10_lamda04)
print(k0_2(lamda,Tm))


# ? Calculo de k0 para tau =0.1 lambda = 0.5
# ? definimos el valor de lambda y del ligando

lamda =0.5
ligand = 0.5


[media_k0_tau10_lamda05, sd_k0_tau10_lamda05 ] = calculo_k0_simulaciones(tau10_lamda05,L,1)

print(media_k0_tau10_lamda05)
print(sd_k0_tau10_lamda05)
print(k0_2(lamda,Tm))

# ? Calculo de k0 para tau =0.1 lambda = 0.7
# ? definimos el valor de lambda y del ligando

lamda =0.7
ligand = 0.5


[media_k0_tau10_lamda07, sd_k0_tau10_lamda07 ] = calculo_k0_simulaciones(tau10_lamda07,L,1)

print(media_k0_tau10_lamda07)
print(sd_k0_tau10_lamda07)
print(k0_2(lamda,Tm))

# ? Calculo de k0 para tau =0.1 lambda = 1.0
# ? definimos el valor de lambda y del ligando

lamda =1.0
ligand = 0.5




[media_k0_tau10_lamda10, sd_k0_tau10_lamda10 ] = calculo_k0_simulaciones(tau10_lamda10,L,1)

print(media_k0_tau10_lamda10)
print(sd_k0_tau10_lamda10)
print(k0_2(lamda,Tm))

# ? Calculo de k0 para tau =0.1 lambda = 1.2
# ? definimos el valor de lambda y del ligando

lamda =1.2
ligand = 0.5


[media_k0_tau10_lamda12, sd_k0_tau10_lamda12 ] = calculo_k0_simulaciones(tau10_lamda12,L,1)

print(media_k0_tau10_lamda12)
print(sd_k0_tau10_lamda12)
print(k0_2(lamda,Tm))

# ? Calculo de k0 para tau =0.1 lambda = 1.5
# ? definimos el valor de lambda y del ligando

lamda =1.5
ligand = 0.5


[media_k0_tau10_lamda15, sd_k0_tau10_lamda15 ] = calculo_k0_simulaciones(tau10_lamda15,L,1)

print(media_k0_tau10_lamda15)
print(sd_k0_tau10_lamda15)
print(k0_2(lamda,Tm))

# ? Calculo de k0 para tau =10.0 lambda = 2.0
# ? definimos el valor de lambda y del ligando

lamda =2.0
ligand = 0.5



[media_k0_tau10_lamda20, sd_k0_tau10_lamda20 ] = calculo_k0_simulaciones(tau10_lamda20,L,1)

print(media_k0_tau10_lamda20)
print(sd_k0_tau10_lamda20)
print(k0_2(lamda,Tm))


# !   ?????????????????????????????????????????  #

# ! Figuras para tau =0.1 l

#! Definimos L, y los vectores longitud





L=10
Tm = 0.1

#! Arreglo espacial de la configuracion 

dx = L/300

xfield = np.arange(0,L,dx)

x_bin = xfield + .5*dx


xx = np.arange(-L/2,L/2,dx)

# ! Calculo de k0 para tau =0.1 lambda = 0.1
# ! definimos el valor de lambda y del ligando

lamda =0.1
ligand = 0.5


[media_k0_tau01_lamda01, sd_k0_tau01_lamda01 ] = calculo_k0_simulaciones(tau01_lamda01,L,1)

print(media_k0_tau01_lamda01)
print(sd_k0_tau01_lamda01)
print(k0_2(lamda,Tm))



# ! Calculo de k0 para tau =0.1 lambda = 0.2
# ! definimos el valor de lambda y del ligando

lamda =0.2
ligand = 0.5


[media_k0_tau01_lamda02, sd_k0_tau01_lamda02 ] = calculo_k0_simulaciones(tau01_lamda02,L,1)

print(media_k0_tau01_lamda02)
print(sd_k0_tau01_lamda02)
print(k0_2(lamda,Tm))


# ! Calculo de k0 para tau =0.1 lambda = 0.4
# ! definimos el valor de lambda y del ligando

lamda =0.4
ligand = 0.5


[media_k0_tau01_lamda04, sd_k0_tau01_lamda04 ] = calculo_k0_simulaciones(tau01_lamda04,L,1)

print(media_k0_tau01_lamda04)
print(sd_k0_tau01_lamda04)
print(k0_2(lamda,Tm))


# ! Calculo de k0 para tau =0.1 lambda = 0.5
# ! definimos el valor de lambda y del ligando

lamda =0.5
ligand = 0.5


[media_k0_tau01_lamda05, sd_k0_tau01_lamda05 ] = calculo_k0_simulaciones(tau01_lamda05,L,1)

print(media_k0_tau01_lamda05)
print(sd_k0_tau01_lamda05)
print(k0_2(lamda,Tm))

# ! Calculo de k0 para tau =0.1 lambda = 0.7
# ! definimos el valor de lambda y del ligando

lamda =0.7
ligand = 0.5


[media_k0_tau01_lamda07, sd_k0_tau01_lamda07 ] = calculo_k0_simulaciones(tau01_lamda07,L,1)

print(media_k0_tau01_lamda07)
print(sd_k0_tau01_lamda07)
print(k0_2(lamda,Tm))

# ! Calculo de k0 para tau =0.1 lambda = 1.0
# ! definimos el valor de lambda y del ligando

lamda =1.0
ligand = 0.5


[media_k0_tau01_lamda10, sd_k0_tau01_lamda10 ] = calculo_k0_simulaciones(tau01_lamda10,L,1)

print(media_k0_tau01_lamda10)
print(sd_k0_tau01_lamda10)
print(k0_2(lamda,Tm))

# ! Calculo de k0 para tau =0.1 lambda = 1.2
# ! definimos el valor de lambda y del ligando

lamda =1.2
ligand = 0.5


[media_k0_tau01_lamda12, sd_k0_tau01_lamda12 ] = calculo_k0_simulaciones(tau01_lamda12,L,1)

print(media_k0_tau01_lamda12)
print(sd_k0_tau01_lamda12)
print(k0_2(lamda,Tm))

# ! Calculo de k0 para tau =0.1 lambda = 1.5
# ! definimos el valor de lambda y del ligando

lamda =1.5
ligand = 0.5


[media_k0_tau01_lamda15, sd_k0_tau01_lamda15 ] = calculo_k0_simulaciones(tau01_lamda15,L,1)

print(media_k0_tau01_lamda15)
print(sd_k0_tau01_lamda15)
print(k0_2(lamda,Tm))

# ! Calculo de k0 para tau =0.1 lambda = 2.0
# ! definimos el valor de lambda y del ligando

lamda =2.0
ligand = 0.5


[media_k0_tau01_lamda20, sd_k0_tau01_lamda20 ] = calculo_k0_simulaciones(tau01_lamda20,L,1)

print(media_k0_tau01_lamda20)
print(sd_k0_tau01_lamda20)
print(k0_2(lamda,Tm))



#  *  ?????????????????????????????????????????  #

# * Figuras para tau =1.0 l

#* Definimos L, y los vectores longitud





L=20
Tm = 1.0

#* Arreglo espacial de la configuracion 

dx = L/300

xfield = np.arange(0,L,dx)

x_bin = xfield + .5*dx


xx = np.arange(-L/2,L/2,dx)

# * Calculo de k0 para tau =0.1 lambda = 0.1
# * definimos el valor de lambda y del ligando

lamda =0.1
ligand = 0.5


[media_k0_tau1_lamda01, sd_k0_tau1_lamda01 ] = calculo_k0_simulaciones(tau1_lamda01,L,1)

print(media_k0_tau1_lamda01)
print(sd_k0_tau1_lamda01)
print(k0_2(lamda,Tm))



# * Calculo de k0 para tau =0.1 lambda = 0.2
# * definimos el valor de lambda y del ligando

lamda =0.2
ligand = 0.5


[media_k0_tau1_lamda02, sd_k0_tau1_lamda02 ] = calculo_k0_simulaciones(tau1_lamda02,L,1)

print(media_k0_tau1_lamda02)
print(sd_k0_tau1_lamda02)
print(k0_2(lamda,Tm))


# * Calculo de k0 para tau =0.1 lambda = 0.4
# * definimos el valor de lambda y del ligando

lamda =0.4
ligand = 0.5


[media_k0_tau1_lamda04, sd_k0_tau1_lamda04 ] = calculo_k0_simulaciones(tau1_lamda04,L,1)

print(media_k0_tau1_lamda04)
print(sd_k0_tau1_lamda04)
print(k0_2(lamda,Tm))


# * Calculo de k0 para tau =0.1 lambda = 0.5
# * definimos el valor de lambda y del ligando

lamda =0.5
ligand = 0.5


[media_k0_tau1_lamda05, sd_k0_tau1_lamda05 ] = calculo_k0_simulaciones(tau1_lamda05,L,1)

print(media_k0_tau1_lamda05)
print(sd_k0_tau1_lamda05)
print(k0_2(lamda,Tm))

# * Calculo de k0 para tau =0.1 lambda = 0.7
# * definimos el valor de lambda y del ligando

lamda =0.7
ligand = 0.5


[media_k0_tau1_lamda07, sd_k0_tau1_lamda07 ] = calculo_k0_simulaciones(tau1_lamda07,L,1)

print(media_k0_tau1_lamda07)
print(sd_k0_tau1_lamda07)
print(k0_2(lamda,Tm))

# * Calculo de k0 para tau =0.1 lambda = 1.0
# * definimos el valor de lambda y del ligando

lamda =1.0
ligand = 0.5


[media_k0_tau1_lamda10, sd_k0_tau1_lamda10 ] = calculo_k0_simulaciones(tau1_lamda10,L,1)

print(media_k0_tau1_lamda10)
print(sd_k0_tau1_lamda10)
print(k0_2(lamda,Tm))

# * Calculo de k0 para tau =0.1 lambda = 1.2
# * definimos el valor de lambda y del ligando

lamda =1.2
ligand = 0.5


[media_k0_tau1_lamda12, sd_k0_tau1_lamda12 ] = calculo_k0_simulaciones(tau1_lamda12,L,1)

print(media_k0_tau1_lamda12)
print(sd_k0_tau1_lamda12)
print(k0_2(lamda,Tm))

# * Calculo de k0 para tau =0.1 lambda = 1.5
# * definimos el valor de lambda y del ligando

lamda =1.5
ligand = 0.5


[media_k0_tau1_lamda15, sd_k0_tau1_lamda15 ] = calculo_k0_simulaciones(tau1_lamda15,L,1)

print(media_k0_tau1_lamda15)
print(sd_k0_tau1_lamda15)
print(k0_2(lamda,Tm))

# * Calculo de k0 para tau =0.1 lambda = 2.0
# * definimos el valor de lambda y del ligando

lamda =2.0
ligand = 0.5


[media_k0_tau1_lamda20, sd_k0_tau1_lamda20 ] = calculo_k0_simulaciones(tau1_lamda20,L,1)

print(media_k0_tau1_lamda20)
print(sd_k0_tau1_lamda20)
print(k0_2(lamda,Tm))




# !! graficas de rho 


L=20
Tm = 1.0
lamda= 1.0
#* Arreglo espacial de la configuracion 

dx = L/300

xfield = np.arange(0,L,dx)

x_bin = xfield + .5*dx


xx = np.arange(-L/2,L/2,dx)


fig1=plt.figure(figsize=(14,8))

# plt.plot(tau01_lamda01[4][0],-tau01_lamda01[4][2],color= "C0",label=r"$\lambda = 0.1,\tau=0.1,l_0=0.1$")
# plt.plot(tau01_lamda01[4][0],tau01_lamda01[4][2]/tau01_lamda01[4][3],color= "C0",label=r"$\lambda = 0.1,\tau=0.1,l_0=0.1$")
# plt.plot(tau1_lamda01[4][0],tau1_lamda01[4][1],color= "C1")
plt.plot(tau1_lamda10[3][0],tau1_lamda10[3][1],color= "C1")
plt.plot(xx,0.53*lorentz_fr(lamda,Tm,xx,L)*ligando(x_bin,0.4,L),color="C0")

# plt.plot(tau01_lamda01[4][0][int(len(tau01_lamda01)/2):],exponential(tau01_lamda01[4][0][int(len(tau01_lamda01)/2):],tau01_lamda01[4][0][int(len(tau01_lamda01)/2):],media_k0_tau01_lamda01),color="C1")
# plt.plot(tau01_lamda01[4][0][int(len(tau01_lamda01)/2):],exponential(tau01_lamda01[4][0][int(len(tau01_lamda01)/2):],popt4[0],k0(lamda,Tm)),color="C3")


plt.xlabel(r'$x$',fontsize=40)

plt.ylabel(r'$\Delta\rho/\rho_0$',fontsize=40)

plt.legend(fontsize=30)

plt.xticks(fontsize=25)
plt.yticks(fontsize=25)



plt.tight_layout()


plt.savefig('response_to_step_function_tau'+str(Tm)+'lambda_'+str(lamda)+'.pdf',dpi=1200)

# ! Gráficas de k0 para distintos tau

k0_medias_2 =[media_k0_tau01_lamda01,media_k0_tau01_lamda02,media_k0_tau01_lamda04,media_k0_tau01_lamda05,media_k0_tau01_lamda07,media_k0_tau01_lamda10,media_k0_tau01_lamda12,media_k0_tau01_lamda15,media_k0_tau01_lamda20]
k0_error_medias_2=[sd_k0_tau01_lamda01,sd_k0_tau01_lamda02,sd_k0_tau01_lamda04,sd_k0_tau01_lamda05,sd_k0_tau01_lamda07,sd_k0_tau01_lamda10,sd_k0_tau01_lamda12,sd_k0_tau01_lamda15,sd_k0_tau01_lamda20]

k0_medias_10 =[media_k0_tau10_lamda01,media_k0_tau10_lamda02,media_k0_tau10_lamda04,media_k0_tau10_lamda05,media_k0_tau10_lamda07,media_k0_tau10_lamda10,media_k0_tau10_lamda12,media_k0_tau10_lamda15,media_k0_tau10_lamda20]
k0_error_medias_10=[sd_k0_tau10_lamda01,sd_k0_tau10_lamda02,sd_k0_tau10_lamda04,sd_k0_tau10_lamda05,sd_k0_tau10_lamda07,sd_k0_tau10_lamda10,sd_k0_tau10_lamda12,sd_k0_tau10_lamda15,sd_k0_tau10_lamda20]


k0_medias_1 =[media_k0_tau1_lamda01,media_k0_tau1_lamda02,media_k0_tau1_lamda04,media_k0_tau1_lamda05,media_k0_tau1_lamda07,media_k0_tau1_lamda10,media_k0_tau1_lamda12,media_k0_tau1_lamda15,media_k0_tau1_lamda20]
k0_error_medias_1=[sd_k0_tau1_lamda01,sd_k0_tau1_lamda02,sd_k0_tau1_lamda04,sd_k0_tau1_lamda05,sd_k0_tau1_lamda07,sd_k0_tau1_lamda10,sd_k0_tau1_lamda12,sd_k0_tau1_lamda15,sd_k0_tau1_lamda20]



lmb_array = [0.1,0.2,0.4,0.5,0.7,1.0,1.2,1.5,2.0]
lmbs = np.linspace(0.0,2.0,100)
# ??? TAU = 0.1 

fig3=plt.figure(figsize=(14,8))

plt.plot(lmbs,k0_2(lmbs,0.1)/((np.sqrt(1+0.1)/0.1)*np.ones(len(lmbs))),color="k")

plt.errorbar(lmb_array,k0_medias_2/((np.sqrt(1+0.1)/0.1)*np.ones(len(lmb_array))), yerr=k0_error_medias_2/((np.sqrt(1+0.1)/0.1)*np.ones(len(lmb_array))),mfc="none",capsize=10,ms=12, color='r',marker="s",linestyle="",label= r" $\tau_{sim} =  0.1$ "    ) 



plt.xlabel(r'$\lambda$',fontsize=30)

plt.ylabel(r'$k_0$',fontsize=30)

plt.legend(fontsize=30,loc="best")

plt.xticks(fontsize=40)
plt.yticks(fontsize=40)



plt.xlim(0.0,2.1)


plt.tight_layout()


# ??? TAU = 10.0 

fig3=plt.figure(figsize=(14,8))

plt.plot(lmbs,k0(lmbs,10.0)/((np.sqrt(1+10)/10)*np.ones(len(lmbs))),color="k")

plt.errorbar(lmb_array,k0_medias_10/((np.sqrt(1+10)/10)*np.ones(len(lmb_array))), yerr=k0_error_medias_10/((np.sqrt(1+10)/10)*np.ones(len(lmb_array))),mfc="none",capsize=10,ms=12, color='r',marker="s",linestyle="",label= r" $\tau_{sim} =  10.0$ "    ) 
# plt.plot(lmbs,(np.sqrt(1+10)/10)*np.ones(len(lmbs)),color="k",linestyle ="--")



plt.xlabel(r'$\lambda$',fontsize=30)

plt.ylabel(r'$k_0$',fontsize=30)

plt.legend(fontsize=30,loc="best")

plt.xticks(fontsize=40)
plt.yticks(fontsize=40)



plt.xlim(0.0,2.1)


plt.tight_layout()



# ??? TAU = 1.0 

fig3=plt.figure(figsize=(14,8))

plt.plot(lmbs,k0(lmbs,1.0)/((np.sqrt(1+1)/1)*np.ones(len(lmbs))),color="k")

plt.errorbar(lmb_array,k0_medias_1/((np.sqrt(1+1)/1)*np.ones(len(lmb_array))), yerr=k0_error_medias_1/((np.sqrt(1+1)/1)*np.ones(len(lmb_array))),mfc="none",capsize=10,ms=12, color='r',marker="s",linestyle="",label= r" $\tau_{sim} =  1.0$ "    ) 
# plt.plot(lmbs,(np.sqrt(1+1)/1)*np.ones(len(lmbs)),color="k",linestyle ="--")



plt.xlabel(r'$\lambda$',fontsize=30)

plt.ylabel(r'$k_0$',fontsize=30)

plt.legend(fontsize=30,loc="best")

plt.xticks(fontsize=40)
plt.yticks(fontsize=40)



plt.xlim(0.0,2.1)


plt.tight_layout()


plt.show()