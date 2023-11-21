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
from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition,
                                                  mark_inset)
import random
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
k = np.linspace(0,k_critico(lamb,tau)+10,1000)
kk = np.linspace(k_critico(lamb,tau),k_critico(lamb,tau)+0.2,100)

print(fun.D_2(lamb,tau))
print(fun.Dx_2(lamb,tau))
print(fun.DDx_2(lamb,tau))

fig3,ax1=plt.subplots(figsize=(14,8))


ax1.plot(k, eigen1(lamb,tau,k),color="k")
ax1.plot(k, k**2*fun.D_2(lamb,tau),color="k",linestyle=":")
# plt.plot(k/k_critico(lamb,tau), fun.D_2(lamb,tau)*  k**2,color="k",linestyle = "dotted")
# plt.plot(kk, eigen1_taugg(lamb,tau,kk),color="k",linestyle="--")

ax1.plot(k, eigen2(lamb,tau,k),color="b",linestyle = "--")
ax1.plot(k,((1/tau)*np.ones(len(k)) + k**2*fun.DDx_2(lamb,tau)),color="b",linestyle=":")
# plt.plot(k/k_critico(lamb,tau), (1/tau)*np.ones(len(k)),color="b",linestyle = "dotted")
# plt.plot(k/k_critico(lamb,tau), aprox_lambda2(lamb,tau,k/k_critico(lamb,tau)),color="b",linestyle = "dashdot")
# plt.plot(kk, eigen2_taugg(lamb,tau,kk),color="b",linestyle="--")



ax1.axvline( k_critico(lamb,tau),color="r",linestyle=":")

ax1.text( k_critico(lamb,tau), -17,  "$k^*$", fontsize=30,color="r")

ax1.set_ylabel( r'$\chi_i$',rotation=0.0,labelpad=30,fontsize=40)

ax1.set_xlabel( r'$k$', fontsize=40)




ax1.tick_params(axis='x', labelsize=25)
ax1.tick_params(axis='y', labelsize=25)




ax1.set_xlim(0.00,max(k))
ax1.set_ylim(-0.02,max(eigen2(lamb,tau,k)))




ax2 = plt.axes([0,0,1,1])


#? Manually set the position and relative size of the inset axes within ax1

#               ?   [positionx,positiony,sizex,sizey]
ip = InsetPosition(ax1, [0.5,0.4,0.4,0.4])


ax2.set_axes_locator(ip)





# ! v_1

ax2.plot(k, eigen1(lamb,tau,k),color="k")
ax2.plot(k, fun.D_2(lamb,tau)*  k**2,color="k",linestyle = "dotted")


# ! v_2

ax2.plot(k, eigen2(lamb,tau,k),color="b",linestyle="--")

ax2.plot(k, (1/tau)*np.ones(len(k)),color="b",linestyle = "dotted")



ax2.axvline( k_critico(lamb,tau),color="r",linestyle=":")



ax2.tick_params(axis='x', labelsize=25)
ax2.tick_params(axis='y', labelsize=25)
# Some ad hoc tweaks.

ax2.set_xlim(0.0,k_critico(lamb,tau)+0.2)
ax2.set_ylim(0.0,eigen2(lamb,tau,k_critico(lamb,tau)+0.2)/(k_critico(lamb,tau)**2*fun.D_2(lamb,tau)))









plt.tight_layout()

plt.savefig("eigenvalues.pdf",dpi=1200)






fig3=plt.figure(figsize=(14,8))

# k = np.linspace(0,1,100)
k = np.linspace(0,k_critico(lamb,tau)+10,1000)
# kk = np.linspace(k_critico(lamb,tau),k_critico(lamb,tau)+0.2,100)
plt.loglog(k, eigen1(lamb,tau,k),color="k")
# plt.loglog(k, fun.D_2(lamb,tau)*  k**2,color="k",linestyle = "dotted")
# plt.loglog(kk, eigen1_taugg(lamb,tau,kk),color="k",linestyle="--")

plt.loglog(k, eigen2(lamb,tau,k),color="b",linestyle="--")
# plt.loglog(k, (1/tau)*np.ones(len(k)),color="b",linestyle = "dotted")
# plt.loglog(k, aprox_lambda2(lamb,tau,k),color="b",linestyle = "dashdot")
# plt.loglog(kk, eigen2_taugg(lamb,tau,kk),color="b",linestyle="--")

plt.loglog(k,(k)**2*1.8,color="g",linestyle="dashdot")


plt.axvline( k_critico(lamb,tau),color="r",linestyle=":")

plt.text(k_critico(lamb,tau),3e-5,  "$k^*$", fontsize=30,color="r")


plt.xlabel(r'$\mathrm{log}(k)$',fontsize=40)

plt.ylabel(r'$\mathrm{log}(\chi_i)$',fontsize=40)

# plt.legend(fontsize=30,loc="best")



plt.xlim(5e-2,10.0)

plt.xticks(fontsize=25)
plt.yticks(fontsize=25)



# plt.ylim(0,k0(2.0,Tm))


plt.tight_layout()
plt.savefig("eigenvalues_loglog.pdf",dpi=1200)






plt.show()