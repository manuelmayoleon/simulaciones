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

from sympy import *

import csv

ba= pd.read_csv("density_lambda1.dat",header=None,sep='\s+',names=["pos","dens"])

ba2= pd.read_csv("density_lamb0.5.dat",header=None,sep='\s+',names=["pos","dens"])


N = 1000 # 2000

# Parametros del sistema

v0 = V = B = D =  1
alpha = 0.5


############## Valores gradiente y tiempo de memoria

Tm = 1

##############

L = 20.0

dx = L/300



# Arreglo espacial de la configuracion

xfield = np.arange(0,L,dx)

x_bin = xfield + .5*dx


xx = np.arange(-L/2,L/2,dx)


def ligando_escalar(x):

#    return math.tanh(4*math.sin(2*math.pi*x/L))

    if(x<L/2):

        return -1

    else:

        return 1


def lorentzian_form(lamb,tau,x):

   return  (1 - (np.sqrt(2) *np.exp(
            lamb**2/2 - (
            np.sqrt((4 + np.exp(lamb**2/2) *  (6 + 8 *lamb**2 + lamb**4) * tau + 
            np.exp(lamb**2)* (2 + 2* lamb**2 + lamb**4) *tau**2)/(
            tau**2 * (2 + np.exp(lamb**2/2) * (1 + lamb**2) * tau)))*  abs(x))/np.sqrt(2))*
            lamb * (2 + np.exp(lamb**2/2) *  (1 + lamb**2) * tau) * np.sqrt((
        4 + np.exp(lamb**2/2) * (6 + 8 * lamb**2 + lamb**4) * tau + 
            np.exp(lamb**2) *  (2 + 2  * lamb**2 + lamb**4) * tau**2)/(
        tau**2 * (2 + np.exp(lamb**2/2)*  (1 + lamb**2) *  tau))))/(
        4 + np.exp(lamb**2/2)  * (6 + 8  * lamb**2 + lamb**4)  * tau + 
        np.exp(lamb**2) *  (2 + 2*  lamb**2 + lamb**4) *  tau**2))




ligando = np.vectorize(ligando_escalar)
    

    
fig1=plt.figure()


plt.plot(xx,ba['dens'],color= "C0")



plt.plot(xx,50+20*lorentzian_form(1,Tm,xx)*ligando(x_bin),color="C1")



plt.xlabel(r'$x$')

plt.ylabel(r'$\rho(x)$')

plt.tight_layout()


plt.xticks(fontsize=20)
plt.yticks(fontsize=20)



fig2=plt.figure()


plt.plot(xx,ba2['dens'],color= "C0")



plt.plot(xx,50+11*lorentzian_form(0.5,Tm,xx)*ligando(x_bin),color="C1")



plt.xlabel('x')

plt.ylabel(r'$\rho(x)$')

plt.tight_layout()


plt.xticks(fontsize=20)
plt.yticks(fontsize=20)





# plt.title ( r' \textbf {Relación $\beta \leftrightarrow \alpha$}' ,fontsize=40)





fig3=plt.figure()


# plt.plot(xx,ba2['dens'],color= "C0")



plt.plot(xx,lorentzian_form(1,Tm,xx)*ligando(x_bin),color="C0",label=r"$\lambda=1$")

plt.plot(xx,lorentzian_form(0.5,Tm,xx)*ligando(x_bin),color="C1",label=r"$\lambda=0.5$")


plt.plot(xx,lorentzian_form(0.1,Tm,xx)*ligando(x_bin),color="C2",label=r"$\lambda=0.1$")


plt.xlabel(r'$x$',fontsize=20)

plt.ylabel(r'$\rho(x)$',fontsize=20)

plt.legend()

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)



plt.tight_layout()

plt.show()