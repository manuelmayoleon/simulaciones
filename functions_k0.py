# functions_k0
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
        return  psi0*(1 - np.exp(- k0* abs(x-np.sign(x)*L*0.5))) /k0
    else:
        return  psi0*(1 - np.exp(- k0* abs(x))) /k0

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



def lorentzian_form_articulo(lamb,tau,x):
    
   return lamb*tau*(1 - np.exp(-np.sqrt(1+tau)*abs(x)/tau))/(1+tau)


ligando = np.vectorize(ligando_escalar)
lorentz_fr = np.vectorize(lorentzian_form)
lorentz_rhox = np.vectorize(resp_rhox)
def lorentz(x,a,b):
   return  a*np.sign(x)*(1 - np.exp(- b* abs(x)))


def exponential(x,a,b):
   return  a* np.exp(- b* x)






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




def calculo_k0_simulaciones_2(vector,L,index):
    j=0
    k0_media_v = np.zeros(len(vector))
    sd_k0_v = np.zeros(len(vector))

    for i in vector:
        tt = obtener_k0_2(i,L,index)
        popt, pcov = curve_fit(lorentz,np.reshape(tt[1], tt[1].size), np.reshape(tt[0], tt[0].size))
        k0_media_v[j] =  popt[1]
        j +=1
        
    k0_media = sum(k0_media_v)/ len(k0_media_v) 
    sd_k0 = np.sqrt(sum(k0_media_v**2)/len(k0_media_v)-k0_media**2)
    
    return [k0_media,sd_k0]


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
