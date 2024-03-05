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
import importlib
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
import sklearn 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

import functions_k0 as funciones



def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


path = os.getcwd()
path = path+"/Promedios/"
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
print(tau1_lamda10[0])



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





[media_k0_tau1_lamda01, sd_k0_tau1_lamda01 ] = funciones.calculo_k0_simulaciones_2(tau1_lamda01,L,1)

print(media_k0_tau1_lamda01)
print(sd_k0_tau1_lamda01)
print(funciones.k0_2(lamda,Tm))



# * Calculo de k0 para tau =0.1 lambda = 0.2
# * definimos el valor de lambda y del ligando

lamda =0.2
ligand = 0.5






[media_k0_tau1_lamda02, sd_k0_tau1_lamda02 ] = funciones.calculo_k0_simulaciones_2(tau1_lamda02,L,1)

print(media_k0_tau1_lamda02)
print(sd_k0_tau1_lamda02)
print(funciones.k0_2(lamda,Tm))


# * Calculo de k0 para tau =0.1 lambda = 0.4
# * definimos el valor de lambda y del ligando

lamda =0.4
ligand = 0.5


[media_k0_tau1_lamda04, sd_k0_tau1_lamda04 ] = funciones.calculo_k0_simulaciones_2(tau1_lamda04,L,1)

print(media_k0_tau1_lamda04)
print(sd_k0_tau1_lamda04)
print(funciones.k0_2(lamda,Tm))


# * Calculo de k0 para tau =0.1 lambda = 0.5
# * definimos el valor de lambda y del ligando

lamda =0.5
ligand = 0.5


[media_k0_tau1_lamda05, sd_k0_tau1_lamda05 ] = funciones.calculo_k0_simulaciones_2(tau1_lamda05,L,1)

print(media_k0_tau1_lamda05)
print(sd_k0_tau1_lamda05)
print(funciones.k0_2(lamda,Tm))

# * Calculo de k0 para tau =0.1 lambda = 0.7
# * definimos el valor de lambda y del ligando

lamda =0.7
ligand = 0.5


[media_k0_tau1_lamda07, sd_k0_tau1_lamda07 ] = funciones.calculo_k0_simulaciones_2(tau1_lamda07,L,1)

print(media_k0_tau1_lamda07)
print(sd_k0_tau1_lamda07)
print(funciones.k0_2(lamda,Tm))

# * Calculo de k0 para tau =0.1 lambda = 1.0
# * definimos el valor de lambda y del ligando

lamda =1.0
ligand = 0.5


[media_k0_tau1_lamda10, sd_k0_tau1_lamda10 ] = funciones.calculo_k0_simulaciones_2(tau1_lamda10,L,1)

print(media_k0_tau1_lamda10)
print(sd_k0_tau1_lamda10)
print(funciones.k0_2(lamda,Tm))

# * Calculo de k0 para tau =0.1 lambda = 1.2
# * definimos el valor de lambda y del ligando

lamda =1.2
ligand = 0.5


[media_k0_tau1_lamda12, sd_k0_tau1_lamda12 ] = funciones.calculo_k0_simulaciones_2(tau1_lamda12,L,1)

print(media_k0_tau1_lamda12)
print(sd_k0_tau1_lamda12)
print(funciones.k0_2(lamda,Tm))

# * Calculo de k0 para tau =0.1 lambda = 1.5
# * definimos el valor de lambda y del ligando

lamda =1.5
ligand = 0.5


[media_k0_tau1_lamda15, sd_k0_tau1_lamda15 ] = funciones.calculo_k0_simulaciones_2(tau1_lamda15,L,1)

print(media_k0_tau1_lamda15)
print(sd_k0_tau1_lamda15)
print(funciones.k0_2(lamda,Tm))

# * Calculo de k0 para tau =0.1 lambda = 2.0
# * definimos el valor de lambda y del ligando

lamda =2.0
ligand = 0.5


[media_k0_tau1_lamda20, sd_k0_tau1_lamda20 ] = funciones.calculo_k0_simulaciones_2(tau1_lamda20,L,1)

print(media_k0_tau1_lamda20)
print(sd_k0_tau1_lamda20)
print(funciones.k0_2(lamda,Tm))


k0_medias_1 =[media_k0_tau1_lamda01,media_k0_tau1_lamda02,media_k0_tau1_lamda04,media_k0_tau1_lamda05,media_k0_tau1_lamda07,media_k0_tau1_lamda10,media_k0_tau1_lamda12,media_k0_tau1_lamda15,media_k0_tau1_lamda20]
k0_error_medias_1=[sd_k0_tau1_lamda01,sd_k0_tau1_lamda02,sd_k0_tau1_lamda04,sd_k0_tau1_lamda05,sd_k0_tau1_lamda07,sd_k0_tau1_lamda10,sd_k0_tau1_lamda12,sd_k0_tau1_lamda15,sd_k0_tau1_lamda20]



lmb_array = [0.1,0.2,0.4,0.5,0.7,1.0,1.2,1.5,2.0]
lmbs = np.linspace(0.0,2.0,100)

# ??? TAU = 1.0 

fig3=plt.figure(figsize=(14,8))

plt.plot(lmbs,funciones.k0(lmbs,1.0),color="k",linewidth=2.0)

plt.errorbar(lmb_array,k0_medias_1, yerr=k0_error_medias_1/((np.sqrt(1+1)/1)*np.ones(len(lmb_array))),mfc="none",capsize=10,ms=12, color='r',marker="s",linestyle="",label= r" $\tau_{sim} =  1.0$ "    ) 
# plt.plot(lmbs,(np.sqrt(1+1)/1)*np.ones(len(lmbs)),color="k",linestyle ="--")



plt.xlabel(r'$\lambda$',fontsize=40)

plt.ylabel(r'$k_0$',fontsize=40)

#plt.legend(fontsize=30,loc="best")

plt.xticks(fontsize=25)
plt.yticks(fontsize=25)



plt.xlim(0.0,2.1)


plt.tight_layout()

plt.savefig('k0_vs_lambda'+str(1)+'_1.pdf',dpi=1200)






fig3,ax1=plt.subplots(figsize=(14,8))


ax1.plot(lmbs,funciones.k0(lmbs,1.0),color="k",linewidth=2.0)
ax1.errorbar(lmb_array,k0_medias_1, yerr=k0_error_medias_1,mfc="none",capsize=10,ms=12, color='r',marker="s",linestyle="",label= r" $\tau_{sim} =  1.0$ "    ) 




ax1.set_ylabel(r'$k_0$',fontsize=40)

ax1.set_xlabel( r'$\lambda$', fontsize=40)




ax1.tick_params(axis='x', labelsize=25)
ax1.tick_params(axis='y', labelsize=25)




ax1.set_xlim(0.00,max(lmbs)+0.1)
ax1.set_ylim(0.00,funciones.k0(2.0,1.0))




ax2 = plt.axes([0,0,1,1])


#? Manually set the position and relative size of the inset axes within ax1

#               ?   [positionx,positiony,sizex,sizey]
ip = InsetPosition(ax1, [0.1,0.5,0.45,0.45])


ax2.set_axes_locator(ip)





ax2.plot(lmbs,funciones.k0(lmbs,1.0),color="k",linewidth=2.0)
ax2.errorbar(lmb_array,k0_medias_1, yerr=k0_error_medias_1,mfc="none",capsize=10,ms=12, color='r',marker="s",linestyle="",label= r" $\tau_{sim} =  1.0$ "    ) 






ax2.tick_params(axis='x', labelsize=25)
ax2.tick_params(axis='y', labelsize=25)
# Some ad hoc tweaks.

ax2.set_xlim(0.0,1.05)
ax2.set_ylim(1.0,2.0)


plt.tight_layout()

plt.savefig('k0_vs_lambda'+str(1)+'_con_ventana_1.pdf',dpi=1200)










fig3=plt.figure(figsize=(14,8))

plt.plot(lmbs,(funciones.k0(lmbs,1.0))**(-1),color="k",linewidth=2.0)

plt.errorbar(lmb_array,1./np.array(k0_medias_1), yerr=(1./np.array(k0_medias_1))*np.array(k0_error_medias_1),mfc="none",capsize=10,ms=12, color='r',marker="s",linestyle="",label= r" $\tau_{sim} =  1.0$ "    ) 
# plt.plot(lmbs,(np.sqrt(1+1)/1)*np.ones(len(lmbs)),color="k",linestyle ="--")



plt.xlabel(r'$\lambda$',fontsize=40)

plt.ylabel(r'$L_s$',fontsize=40)

#plt.legend(fontsize=30,loc="best")

plt.xticks(fontsize=25)
plt.yticks(fontsize=25)



plt.xlim(0.0,2.1)


plt.tight_layout()

plt.savefig('Ls_vs_lambda'+str(1)+'.pdf',dpi=1200)



plt.show()