# homogeneous_stationary.py
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

from sklearn.linear_model import LinearRegression


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def open_files(name):
    with open(name) as csvfile:
        rox0_stat = csv.reader(csvfile, delimiter=',')
        # print(rox0_stat)
        for row in rox0_stat:
            rox =  row
      
    roxx = [float(s) for s in rox]
    
    return roxx 
    

# with open("rox_stat.csv") as csvfile:
#     rox0_stat = csv.reader(csvfile, delimiter=',')
#     for row in rox0_stat:
#       rox =  row
# roxx = [float(s) for s in rox]
# with open("rhox0_stat_l05.dat") as csvfile:
#     rox0_stat = csv.reader(csvfile, delimiter=',')
#     for row in rox0_stat:
#       rox =  row
# roxx05 = [float(s) for s in rox]

# rox0_stat05= pd.read_csv("rhox0_stat_l05.dat",header=None,sep=',',names=["rhox"])

# roxx05 = [float(s) for s in rox0_stat05]

roxx = open_files("rox_stat.csv")
roxx05 = open_files("rhox0_stat_l05.dat")
roxx02 = open_files("rhox0_stat_l02.dat")
roxx15 = open_files("rhox0_stat_l15.dat")
roxx3 = open_files("rhox0_stat_l3.dat")

j02 = open_files("jstat_l02.dat")
j05 = open_files("jstat_l05.dat")
j1 = open_files("jstat_l1.dat")
j15 = open_files("jstat_l15.dat")
j3 = open_files("jstat_l3.dat")

print(np.array(roxx))

print(roxx05)

tau=np.linspace(0,100,len(roxx))

print(tau)
    
fig1=plt.figure(figsize=(14,8))
plt.plot(tau,np.array(roxx3),color= "k",linestyle = "dashdot",label=r"$l_0 = 3$",linewidth=2.0)
# plt.plot(tau,np.array(roxx15),color= "C3",linestyle = "dotted",label=r"$l_0 = 1.5$")
plt.plot(tau,np.array(roxx),color= "b",linestyle = "-" ,label=r"$l_0 = 1$",linewidth=2.0)
plt.plot(tau,np.array(roxx05),color= "r",linestyle = "--",label=r"$l_0 = 0.5$",linewidth=2.0)
# plt.plot(tau,np.array(roxx02),color= "C2",linestyle = "dashdot",label=r"$l_0 = 0.2$")
# plt.plot(tau,np.array(roxx15),color= "C3",linestyle = "dotted",label=r"$l_0 = 1.5$")




plt.xlabel(r'$\tau\nu_0^{-1}$',fontsize=40)

plt.ylabel(r'$\rho_{X}/\rho$',fontsize=40)

# plt.legend(fontsize=25)

plt.xticks(fontsize=25)
plt.yticks(fontsize=25)

plt.xlim(0,6)





plt.tight_layout()

plt.savefig("stationary_rhox0_homogeneous.pdf",dpi=1200)


  
fig2=plt.figure(figsize=(14,8))


plt.plot(tau,np.array(j3),color= "k",linestyle = "dashdot",label=r"$l_0 = 3$",linewidth=2.0)
# plt.plot(tau,np.array(j15),color= "C3",linestyle = "dotted",label=r"$l_0 = 1.5$")
plt.plot(tau,np.array(j1),color= "b",linestyle = "-" ,label=r"$l_0 = 1$",linewidth=2.0)
plt.plot(tau,np.array(j05),color= "r",linestyle = "--",label=r"$l_0 = 0.5$",linewidth=2.0)
# plt.plot(tau,np.array(j02),color= "C2",linestyle = "dashdot",label=r"$l_0 = 0.2$")
# plt.plot(tau,np.array(roxx15),color= "C3",linestyle = "dotted",label=r"$l_0 = 1.5$")




plt.xlabel(r'$\tau\nu_0^{-1}$',fontsize=40)

plt.ylabel(r'$J/(\rho V)$',fontsize=40)

# plt.legend(fontsize=25)

plt.xticks(fontsize=25)
plt.yticks(fontsize=25)

plt.xlim(0,6)





plt.tight_layout()

plt.savefig("stationary_j_homogeneous.pdf",dpi=1200)

plt.show()