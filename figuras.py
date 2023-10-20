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
        # name = file_name.split(".")[0]
        name=file_name
        # print('Name:', name)
        names.append(name)
          
        # print the content
        # print('Content:')
        # print(df)  
# print(elements) 
# print(names)
r01 =  elements[names.index('NormDensidadEscalon-1.0-0.1-0.4.dat')]   
r05 =  elements[names.index('NormDensidadEscalon-1.0-0.5-0.4.dat')]
r1 =  elements[names.index('NormDensidadEscalon-1.0-1.0-0.4.dat')]    
r2 =  elements[names.index('NormDensidadEscalon-1.0-2.0-0.4.dat')]  


ro1= pd.read_csv("density_lambda1.dat",header=None,sep='\s+',names=["pos","dens"])
ro02= pd.read_table("density_lambda0.2_.dat",header=None,sep='\s+',names=["pos","dens","error"])
ro04= pd.read_table("density_lambda0.4_.dat",header=None,sep='\s+',names=["pos","dens","error"])
ro05= pd.read_csv("density_lamb0.5_1.dat",header=None,sep='\s+',names=["pos","dens","errror"])
ro07= pd.read_table("density_lambda0.7_.dat",header=None,sep='\s+',names=["pos","dens","error"])
ro12= pd.read_table("density_lambda1.2_.dat",header=None,sep='\s+',names=["pos","dens"])
ro15= pd.read_table("density_lambda1.5_.dat",header=None,sep='\s+',names=["pos","dens"])


# print(np.transpose(ro02))


error02 =np.sqrt((ro02['dens']-sum(ro02['dens']))**2/10000000)

error04 =np.sqrt((ro04['dens']-sum(ro04['dens']))**2/10000000)





N = 1000 # 2000

# Parametros del sistema

v0 = V = B = D =  1
alpha = 0.5


############## Valores gradiente y tiempo de memoria

Tm = 1.0

##############

L = 20.0

dx = L/300

# Arreglo espacial de la configuracion

xfield = np.arange(0,L,dx)

x_bin = xfield + .5*dx


xx = np.arange(-L/2,L/2,dx)





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


# print(r1[0:])


r1_p = ro_pp(r1[0:])
# print(r1_p)

xxx = np.linspace(-L/4,L/4,len(r1_p))


plt.plot(xxx, r1_p)

popt, pcov = curve_fit(lorentz, xxx, np.reshape(r1_p, r1_p.size))

print(popt[0])
plt.plot(xxx, lorentz(xxx,popt[0],popt[1]))


print(popt[1])



r05_p = ro_pp(r05[0:])
# print(r1_p)



# plt.plot(xxx, r05_p)

popt2, pcov2 = curve_fit(lorentz, xxx, np.reshape(r05_p, r05_p.size))

print(popt2)

# print(1/np.sqrt(1+np.exp(0.5**2/2)*1*(1+0.5**2) ))

ro1_prima = ro_prime(ro1)
# print(ro1_prima)

ro02_prima = ro_prime(ro02)
# print(ro02_prima)

ro04_prima = ro_prime(ro04)
# print(ro04_prima)



ro05_prima = ro_prime(ro05)
# print(ro05_prima)


ro07_prima = ro_prime(ro07)
# print(ro07_prima)


ro12_prima = ro_prime(ro12)
# print(ro12_prima)

xxx = np.linspace(0,L/4,len(ro1_prima))

log_ro1_normalized =  np.log((ro1_prima-ro1_prima[0])/(ro1_prima[0]-ro1_prima[-1])+1)

df = pd.DataFrame( { 'x': np.log(xxx),
                    'y': log_ro1_normalized})

new_ro1_norm  = df[np.isfinite(df).all(1)]

# print(np.array(new_ro1_norm['x']))



model = LinearRegression().fit(np.array(new_ro1_norm['x']).reshape(-1,1)[0:3], new_ro1_norm['y'][0:3])
r_sq = model.score(np.array(new_ro1_norm['x']).reshape(-1,1)[0:3], new_ro1_norm['y'][0:3])

inferred1= model.predict(np.array(new_ro1_norm['x']).reshape(-1,1)[0:3])

#! >>> mean_absolute_error(y_true, y_pred)
model_error1 = mean_absolute_error(new_ro1_norm['y'][0:3], inferred1)


print(f"coefficient of determination: {r_sq}")


print(f"intercept: {model.intercept_}")

print(f"slope: {model.coef_}")

print('model error:',model_error1)


intercept1= model.intercept_
slope1 = model.coef_

y = intercept1 + slope1*np.array(new_ro1_norm['x'])

yy = intercept1 - 0.48*np.array(new_ro1_norm['x'])


print("longitud crítica lambda 1 ",abs(1/(slope1*4)))
print("Error longitud crítica lambda 1 ",(abs(1/(slope1)))**2*model_error1)

print("longitud  teorica lambda 1 ",l_c_teorica(1.0,Tm))



log_ro02_normalized =  np.log((ro02_prima-ro02_prima[0])/(ro02_prima[0]-ro02_prima[-1])+1)

df = pd.DataFrame( { 'x': np.log(xxx),
                    'y': log_ro02_normalized})

new_ro02_norm  = df[np.isfinite(df).all(1)]

# print(np.array(new_ro02_norm['x']))



model = LinearRegression().fit(np.array(new_ro02_norm['x']).reshape(-1,1)[0:3], new_ro02_norm['y'][0:3])


r_sq = model.score(np.array(new_ro02_norm['x']).reshape(-1,1)[0:3], new_ro02_norm['y'][0:3])



inferred02= model.predict(np.array(new_ro02_norm['x']).reshape(-1,1)[0:3])

#! >>> mean_absolute_error(y_true, y_pred)
model_error02 = mean_absolute_error(new_ro02_norm['y'][0:3], inferred02)


print(f"coefficient of determination: {r_sq}")


print(f"intercept: {model.intercept_}")

print(f"slope: {model.coef_}")
print('model error:',model_error02)






intercept02= model.intercept_
slope02 = model.coef_

y02 = intercept02 + slope02*np.array(new_ro02_norm['x'])


print("longitud crítica lambda 0.2 ",abs(1/(slope02*4)))
print("Error longitud crítica lambda 0.2 ",(abs(1/(slope02)))**2*model_error02)


print("longitud  teorica lambda 0.2 ",l_c_teorica(0.2,Tm))



log_ro04_normalized =  np.log((ro04_prima-ro04_prima[0])/(ro04_prima[0]-ro04_prima[-1])+1)

df = pd.DataFrame( { 'x': np.log(xxx),
                    'y': log_ro04_normalized})

new_ro04_norm  = df[np.isfinite(df).all(1)]

# print(np.array(new_ro04_norm['x']))



model = LinearRegression().fit(np.array(new_ro04_norm['x']).reshape(-1,1)[0:4], new_ro04_norm['y'][0:4])


r_sq = model.score(np.array(new_ro04_norm['x']).reshape(-1,1)[0:4], new_ro04_norm['y'][0:4])


inferred04= model.predict(np.array(new_ro04_norm['x']).reshape(-1,1)[0:4])

#! >>> mean_absolute_error(y_true, y_pred)
model_error04 = mean_absolute_error(new_ro04_norm['y'][0:4], inferred04)





print(f"coefficient of determination: {r_sq}")


print(f"intercept: {model.intercept_}")

print(f"slope: {model.coef_}")
print('model error:',model_error04)



intercept04= model.intercept_
slope04 = model.coef_

y02 = intercept04 + slope04*np.array(new_ro04_norm['x'])


print("longitud crítica lambda 0.4 ",abs(1/(slope04*4)))

print("Error longitud crítica lambda 04 ",(abs(1/(slope04)))**2*model_error04)


print("longitud  teorica lambda 0.4 ",l_c_teorica(0.4,Tm))






log_ro05_normalized =  np.log((ro05_prima-ro05_prima[0])/(ro05_prima[0]-ro05_prima[-1])+1)

df = pd.DataFrame( { 'x': np.log(xxx),
                    'y': log_ro05_normalized})

new_ro05_norm  = df[np.isfinite(df).all(1)]

# print(np.array(new_ro05_norm['x']))



model = LinearRegression().fit(np.array(new_ro05_norm['x']).reshape(-1,1)[0:3], new_ro05_norm['y'][0:3])


r_sq = model.score(np.array(new_ro05_norm['x']).reshape(-1,1)[0:3], new_ro05_norm['y'][0:3])


inferred05= model.predict(np.array(new_ro05_norm['x']).reshape(-1,1)[0:3])

#! >>> mean_absolute_error(y_true, y_pred)
model_error05 = mean_absolute_error(new_ro05_norm['y'][0:3], inferred05)



print(f"coefficient of determination: {r_sq}")


print(f"intercept: {model.intercept_}")

print(f"slope: {model.coef_}")
print('model error:',model_error05)


intercept05= model.intercept_
slope05 = model.coef_

y05 = intercept05 + slope05*np.array(new_ro05_norm['x'])


print("longitud crítica lambda 0.5 ",abs(1/(slope05*4)))
print("Error longitud crítica lambda 0.5 ",(abs(1/(slope05)))**2*model_error05)


print("longitud  teorica lambda 0.5 ",l_c_teorica(0.5,Tm))



log_ro07_normalized =  np.log((ro07_prima-ro07_prima[0])/(ro07_prima[0]-ro07_prima[-1])+1)

df = pd.DataFrame( { 'x': np.log(xxx),
                    'y': log_ro07_normalized})

new_ro07_norm  = df[np.isfinite(df).all(1)]

# print(np.array(new_ro07_norm['x']))



model = LinearRegression().fit(np.array(new_ro07_norm['x']).reshape(-1,1)[0:3], new_ro07_norm['y'][0:3])


r_sq = model.score(np.array(new_ro07_norm['x']).reshape(-1,1)[0:3], new_ro07_norm['y'][0:3])

inferred07= model.predict(np.array(new_ro07_norm['x']).reshape(-1,1)[0:3])

#! >>> mean_absolute_error(y_true, y_pred)
model_error07 = mean_absolute_error(new_ro07_norm['y'][0:3], inferred07)


print(f"coefficient of determination: {r_sq}")


print(f"intercept: {model.intercept_}")

print(f"slope: {model.coef_}")

print('model error:',model_error07)



intercept07= model.intercept_
slope07 = model.coef_

y = intercept07 + slope07*np.array(new_ro07_norm['x'])

yy = intercept07 - 0.48*np.array(new_ro07_norm['x'])


print("longitud crítica lambda 0.7 ",abs(1/(slope07*4)))
print("Error longitud crítica lambda 0.7 ",(abs(1/(slope07)))**2*model_error07)

print("longitud  teorica lambda 0.7 ",l_c_teorica(0.7,Tm))








log_ro12_normalized =  np.log((ro12_prima-ro12_prima[0])/(ro12_prima[0]-ro12_prima[-1])+1)

df = pd.DataFrame( { 'x': np.log(xxx),
                    'y': log_ro12_normalized})

new_ro12_norm  = df[np.isfinite(df).all(1)]

# print(np.array(new_ro12_norm['x']))



model = LinearRegression().fit(np.array(new_ro12_norm['x']).reshape(-1,1)[0:3], new_ro12_norm['y'][0:3])


r_sq = model.score(np.array(new_ro12_norm['x']).reshape(-1,1)[0:3], new_ro12_norm['y'][0:3])



inferred12= model.predict(np.array(new_ro12_norm['x']).reshape(-1,1)[0:3])

#! >>> mean_absolute_error(y_true, y_pred)
model_error12 = mean_absolute_error(new_ro12_norm['y'][0:3], inferred12)


print(f"coefficient of determination: {r_sq}")


print(f"intercept: {model.intercept_}")

print(f"slope: {model.coef_}")
print('model error:',model_error12)



intercept12= model.intercept_
slope12 = model.coef_

y12 = intercept12 + slope12*np.array(new_ro12_norm['x'])


print("longitud crítica lambda 1.2 ",abs(1/(slope12*4)))
print("Error longitud crítica lambda 1.2 ",(abs(1/(slope12)))**2*model_error12)


print("longitud  teorica lambda 1.2 ",l_c_teorica(1.2,Tm))



def ligando_escalar(x):

#    return math.tanh(4*math.sin(2*math.pi*x/L))

    if(x<L/2):

        return -0.4

    else:

        return 0.4


def lorentzian_form(lamb,tau,x):
   psi0=lamb*np.exp(lamb**2/2)/(1+np.exp(lamb**2/2)*tau*(1+lamb**2))
   print(psi0*0.4*0.5)
   k0 = np.sqrt((1+np.exp(lamb**2/2)*tau*(1+lamb**2))/tau**2)
   return  psi0*(1 - np.exp(- k0* abs(x)))


def lorentzian_form_articulo(lamb,tau,x):
    
   return lamb*tau*(1 - np.exp(-np.sqrt(1+tau)*abs(x)/tau))/(1+tau)





ligando = np.vectorize(ligando_escalar)
    

    
fig1=plt.figure()


# plt.plot(xx,ro1['dens'],color= "C0",label=r"$\lambda = 1$")

plt.plot(r1[0:][0],r1[0:][1],color= "C0",label=r"$\lambda = 1$")



plt.plot(xx,0.5*lorentzian_form(1,Tm,xx)*ligando(x_bin),color="C1")

# plt.plot(xx,0.5*lorentzian_form_articulo(1,Tm,xx)*ligando(x_bin),color="C1",linestyle="--")
plt.plot(xxx,lorentz(xxx,popt[0],popt[1]),color="C1",linestyle="--")


plt.xlabel(r'$x$')

plt.ylabel(r'$\rho(x)$')

plt.legend()

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)



plt.tight_layout()


   
fig3=plt.figure()


plt.plot(r01[0:][0],r01[0:][1],color= "C2",label=r"$\lambda = 0.1$")



plt.plot(xx,0.5*lorentzian_form(0.1,Tm,xx)*ligando(x_bin),color="C3")

plt.plot(xx,0.5*lorentzian_form_articulo(0.1,Tm,xx)*ligando(x_bin),color="C3",linestyle="--")



plt.xlabel(r'$x$')

plt.ylabel(r'$\rho(x)$')

plt.legend()

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.tight_layout()







   
fig34=plt.figure()


plt.plot(r05[0:][0],r05[0:][1],color= "C4",label=r"$\lambda = 0.5$")



plt.plot(xx,0.5*lorentzian_form(0.5,Tm,xx)*ligando(x_bin),color="C5")

plt.plot(xx,0.5*lorentzian_form_articulo(0.5,Tm,xx)*ligando(x_bin),color="C5",linestyle="--")


plt.xlabel(r'$x$')

plt.ylabel(r'$\rho(x)$')

plt.legend()

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.tight_layout()










# fig2=plt.figure()


# plt.plot(xx,ro05['dens'],color= "C2",label=r"$\lambda = 0.5$")



# plt.plot(xx,0.5*(max(ro05['dens'])+min(ro05['dens']))+np.sqrt(np.pi*0.5)*0.5*(max(ro05['dens'])-min(ro05['dens']))*lorentzian_form(0.5,Tm,xx)*ligando(x_bin),color="C3")

# plt.plot(xx,0.5*(max(ro05['dens'])+min(ro05['dens']))+np.sqrt(np.pi*0.5)*0.5*(max(ro05['dens'])-min(ro05['dens']))*lorentzian_form_articulo(0.5,Tm,xx)*ligando(x_bin),color="C3",linestyle="--")


# plt.xlabel('x')

# plt.ylabel(r'$\rho(x)$')

# plt.legend()

# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)


# plt.tight_layout()






# fig31=plt.figure()


# plt.plot(xx,ro07['dens'],color= "C6",label=r"$\lambda = 0.7$")



# plt.plot(xx,0.5*(max(ro07['dens'])+min(ro07['dens']))+np.sqrt(np.pi*0.5)*0.5*(max(ro07['dens'])-min(ro07['dens']))*lorentzian_form(0.7,Tm,xx)*ligando(x_bin),color="C7")

# plt.plot(xx,0.5*(max(ro07['dens'])+min(ro07['dens']))+np.sqrt(np.pi*0.5)*0.5*(max(ro07['dens'])-min(ro07['dens']))*lorentzian_form_articulo(0.7,Tm,xx)*ligando(x_bin),color="C7",linestyle="--")


# plt.xlabel(r'$x$')

# plt.ylabel(r'$\rho(x)$')

# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)

# plt.legend()

# plt.tight_layout()







fig24=plt.figure()


# plt.plot(xx,ro05['dens'],color= "C2")



# plt.plot(xxx,(ro_prima-ro_prima[0])/(ro_prima[0]-ro_prima[-1])+1,color="C3")


# plt.plot(new_ro1_norm['x'],new_ro1_norm['y'],color="C3",linestyle="",marker="o")


# plt.plot(np.array(new_ro1_norm['x']),y,color="C4")

# plt.plot(np.array(new_ro1_norm['x']),yy,color="C5")


plt.plot(new_ro05_norm['x'],new_ro05_norm['y'],color="C6",linestyle="",marker="o")


plt.plot(np.array(new_ro05_norm['x']),y05,color="C7")

# plt.plot(np.array(new_ro05_norm['x']),yy05,color="C8")

# plt.yscale('log')



plt.xlabel(r'$x$')

plt.ylabel(r'$\rho(x)$')


plt.legend()


plt.tight_layout()


plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

# plt.title ( r' \textbf {Relación $\beta \leftrightarrow \alpha$}' ,fontsize=40)







fig3=plt.figure()






plt.plot(xx,lorentzian_form(1,Tm,xx)*ligando(x_bin),color="C0",label=r"$\lambda=1$")

plt.plot(xx,lorentzian_form_articulo(1,Tm,xx)*ligando(x_bin),color="C0",linestyle="--",label=r"$\lambda=1$")

plt.plot(xx,lorentzian_form(0.5,Tm,xx)*ligando(x_bin),color="C1",label=r"$\lambda=0.5$")

plt.plot(xx,lorentzian_form_articulo(0.5,Tm,xx)*ligando(x_bin),color="C1",linestyle="--",label=r"$\lambda=0.5$")


plt.plot(xx,lorentzian_form(0.1,Tm,xx)*ligando(x_bin),color="C2",label=r"$\lambda=0.1$")

plt.plot(xx,lorentzian_form_articulo(0.1,Tm,xx)*ligando(x_bin),color="C2",linestyle="--",label=r"$\lambda=0.1$")


plt.xlabel(r'$x$',fontsize=20)

plt.ylabel(r'$\rho(x)$',fontsize=20)

plt.legend()

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)



plt.tight_layout()



fig3=plt.figure()


lmbs=np.linspace(0,1.2,100)
lmb_array= [0.2,0.4,0.7,1.0]
long_array=[abs(1/(slope02*4)),abs(1/(slope04*4)),abs(1/(slope07*4)),abs(1/(slope1*4))]
long_error_array=[float(abs(1/(slope02))**2*model_error02),float(abs(1/(slope04))**2*model_error04),float(abs(1/(slope07))**2*model_error07),float(abs(1/(slope1))**2*model_error1)]

print(long_error_array)

plt.plot(lmbs,l_c_teorica(lmbs,Tm),color="C0",label=r"$\lambda=1$")
plt.errorbar(lmb_array,long_array, yerr=long_error_array,mfc="none",capsize=10,ms=12, color='k',marker="^",linestyle="",label= r" $H =  %2i  \sigma $ "   % 14) 

plt.xlabel(r'$\lambda$',fontsize=20)

plt.ylabel(r'$l_c$',fontsize=20)

# plt.legend()

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)


plt.show()