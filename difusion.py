#  difusion
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
import functions as fun

lamb=1.62
tau = 5


def eigen1(lamb,tau,k):
    return 0.5*(fun.D_2(lamb,tau)*  k**2 + fun.DDx_2(lamb,tau) * k**2 + 1/tau - np.sqrt(
   fun.D_2(lamb,tau)**2 * k**4 + 4 * fun.Dx_2(lamb,tau)**2 * k**4- 2 * fun.D_2(lamb,tau) * fun.DDx_2(lamb,tau) *  k**4+ fun.DDx_2(lamb,tau)**2 * k**4- 2 *  fun.D_2(lamb,tau)* k**2 * 1/tau + 
    2 * fun.DDx_2(lamb,tau) * k**2  * 1/tau + 1/tau**2))
def eigen11(lamb,tau,k):
    return (1 + fun.D_2(lamb,tau) *  k**2 * tau + 
        fun.DDx_2(lamb,tau) *  k**2  * tau - np.sqrt((-1 - fun.D_2(lamb,tau) *  k**2 * tau - fun.DDx_2(lamb,tau)*  k**2 * tau)**2 - 
        4 * (fun.D_2(lamb,tau) *  k**2 *  tau - fun.Dx_2(lamb,tau) *  k**4 * tau**2 + fun.D_2(lamb,tau) *  fun.DDx_2(lamb,tau) * k**4 * tau**2)))/(2 * tau)
def eigen2(lamb,tau,k):
    return 0.5*(fun.D_2(lamb,tau)*  k**2 + fun.DDx_2(lamb,tau) * k**2 + 1/tau + np.sqrt(
   fun.D_2(lamb,tau)**2 * k**4 + 4 * fun.Dx_2(lamb,tau)**2 * k**4- 2 * fun.D_2(lamb,tau) * fun.DDx_2(lamb,tau) *  k**4+ fun.DDx_2(lamb,tau)**2 * k**4- 2 *  fun.D_2(lamb,tau)* k**2 * 1/tau + 
    2 * fun.DDx_2(lamb,tau) * k**2  * 1/tau + 1/tau**2))
def k_critico(lamb,tau):
    return 1/np.sqrt(fun.Dx_2(lamb,tau)*tau) 
def aprox_lambda2(lamb,tau,k):
    return  (1/tau)*np.ones(len(k)) + fun.DDx_2(lamb,tau)*k**2
def eigen1_taugg(lamb,tau,k):
    return 0.5 * (fun.D_2(lamb,tau) + fun.DDx_2(lamb,tau)  - np.sqrt(fun.D_2(lamb,tau)**2 + 4 * fun.Dx_2(lamb,tau) **2 - 2  * fun.D_2(lamb,tau) *  fun.DDx_2(lamb,tau)  + fun.DDx_2(lamb,tau) **2))  * k**2
def eigen2_taugg(lamb,tau,k):
    return 0.5 * (fun.D_2(lamb,tau) + fun.DDx_2(lamb,tau)  + np.sqrt(fun.D_2(lamb,tau)**2 + 4 * fun.Dx_2(lamb,tau) **2 - 2  * fun.D_2(lamb,tau) *  fun.DDx_2(lamb,tau)  + fun.DDx_2(lamb,tau) **2))  * k**2



# k = np.linspace(0,1,100)
k = np.linspace(0,k_critico(lamb,tau)+0.1,100)
kk = np.linspace(k_critico(lamb,tau),k_critico(lamb,tau)+0.2,100)

print(fun.D_2(lamb,tau))
print(fun.Dx_2(lamb,tau))
print(fun.DDx_2(lamb,tau))

fig3=plt.figure(figsize=(14,8))




plt.plot(k, eigen1(lamb,tau,k),color="k")
plt.plot(k, eigen11(lamb,tau,k),color="k")
plt.plot(k, fun.D_2(lamb,tau)*  k**2,color="k",linestyle = "dotted")
plt.plot(kk, eigen1_taugg(lamb,tau,kk),color="k",linestyle="--")

plt.plot(k, eigen2(lamb,tau,k),color="b")
plt.plot(k, (1/tau)*np.ones(len(k)),color="b",linestyle = "dotted")
plt.plot(k, aprox_lambda2(lamb,tau,k),color="b",linestyle = "dashdot")
plt.plot(kk, eigen2_taugg(lamb,tau,kk),color="b",linestyle="--")



plt.axvline( k_critico(lamb,tau),color="r",linestyle="--")



plt.xlabel(r'$k$',fontsize=30)

plt.ylabel(r'$v_i$',fontsize=30)

# plt.legend(fontsize=30,loc="best")

plt.xticks(fontsize=40)
plt.yticks(fontsize=40)



# plt.xlim(0.0,1.0)
# plt.ylim(0,k0(2.0,Tm))


plt.tight_layout()
plt.savefig("eigenvalues.pdf",dpi=1200)

fig3=plt.figure(figsize=(14,8))

# k = np.linspace(0,1,100)
k = np.linspace(0,k_critico(lamb,tau)+5.0,100)
# kk = np.linspace(k_critico(lamb,tau),k_critico(lamb,tau)+0.2,100)
plt.loglog(k, eigen1(lamb,tau,k),color="k")
# plt.loglog(k, fun.D_2(lamb,tau)*  k**2,color="k",linestyle = "dotted")
# plt.loglog(kk, eigen1_taugg(lamb,tau,kk),color="k",linestyle="--")

plt.loglog(k, eigen2(lamb,tau,k),color="b")
# plt.loglog(k, (1/tau)*np.ones(len(k)),color="b",linestyle = "dotted")
# plt.loglog(k, aprox_lambda2(lamb,tau,k),color="b",linestyle = "dashdot")
# plt.loglog(kk, eigen2_taugg(lamb,tau,kk),color="b",linestyle="--")

plt.loglog(k,k**2*1.5,color="g",linestyle=":")


plt.axvline( k_critico(lamb,tau),color="r",linestyle="--")



plt.xlabel(r'$\mathrm{log}k$',fontsize=30)

plt.ylabel(r'$\mathrm{log}v_i$',fontsize=30)

# plt.legend(fontsize=30,loc="best")



plt.xlim(1e-1,5.0)

plt.xticks(fontsize=30)
plt.yticks(fontsize=30)



# plt.ylim(0,k0(2.0,Tm))


plt.tight_layout()
plt.savefig("eigenvalues_loglog.pdf",dpi=1200)






plt.show()