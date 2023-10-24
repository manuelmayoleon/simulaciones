# !!!!!!!!!!!! FUNCIONES !!!
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
def ligando_escalar(x,l0,L):
    
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
    # return   (lamb*  tau *  (2 + np.exp(lamb**2/2)  * (1 + lamb**2) *  tau))/(2 + np.exp(lamb**2/2) *  (3 + 2 * lamb**2) *  tau + np.exp(lamb**2) * tau**2)

    return            (lamb * tau *(2 + np.exp(lamb**2/2) *(1 + lamb**2)* tau))/(2 + 
                            np.exp(lamb**2/2)* (3 + 2* lamb**2)* tau + np.exp(lamb**2)* tau**2)

def DDx_2(lamb,tau):
    return ( tau *  (2 + np.exp(lamb**2/2)  * (1 + 2* lamb**2) *  tau))/(2 + np.exp(lamb**2/2) *  (3 + 2 * lamb**2) *  tau + np.exp(lamb**2) * tau**2)
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

def ro_prime(vector,L,xx):
    
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


def calculo_k0_simulaciones(vector1,vector2,vector3,vector4,vector5,vector6,L):
    
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
