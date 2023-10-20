# import pandas as pd

# import seaborn as sns

# from scipy.optimize import curve_fit

import time

# import winsound

# import os

import math

import numpy as np

import matplotlib.pyplot as plt

import numpy.random as rand

from collections import Counter

# from matplotlib import colors

import matplotlib_inline

# matplotlib_inline.backend_inline.set_matplotlib_formats('svg')

# import matplotlib.animation as animation

import csv 

import os 

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

     

# Paso de tiempo delta t

dt = .0001

# Cantidad de bacterias que simulamos

N = 1000 # 2000

# Parametros del sistema

v0 = V = B = D =  1
alpha = 0.2


############## Valores gradiente y tiempo de memoria

c1 = 5

c0 = 2*c1

Tm = 1

##############



# Tiempo hasta el que se simula

tsim = 10000

# Intervalo de tiempo de la simulacion

t = np.arange(0, tsim, dt)

# Cada cuantos pasos dt almacenamos los datos

cut = 1000


# Largo de la caja donde simulamos

L = 20.0

dx = L/300



# Arreglo espacial de la configuracion

xfield = np.arange(0,L,dx)

x_bin = xfield + .5*dx



# Condiciones iniciales para las bacterias

xi = rand.uniform(0,L,size=N)

yi = np.zeros(N)

# Posicion bacterias

ri = np.array([xi, yi])



# Concentracion inicial de protei­na en las bacterias

Xi = rand.normal(0, 1, size=N)

# Orientacion inicial de las bacterias

thetai = rand.uniform(0, 2*np.pi, size=N)



# valor del ligando en el paso previo

ligandoprevio=ligando(xi)



# Arreglos para almacenar los datos

x = np.zeros([int(len(t)/cut), N])

y = np.zeros([int(len(t)/cut), N])

X = np.zeros([int(len(t)/cut), N])

theta = np.zeros([int(len(t)/cut), N])



# Contador para almacenar los datos

k1 = 0

# Contador para los campos

k2 = 0



# Arreglos para almacenar la densidades acumuladas

rho_acum = np.zeros([len(xfield)])

rhox_acum = np.zeros(len(xfield))

Q1_acum = np.zeros([len(xfield)])

Q2_acum = np.zeros([len(xfield)])

JX1_acum = np.zeros([len(xfield)])

JX2_acum = np.zeros([len(xfield)])

Jx_acum = np.zeros([len(xfield)])

Jy_acum = np.zeros([len(xfield)])

xx = np.arange(-L/2,L/2,dx)







# Para ver tiempo de ejecucion

start = time.perf_counter()






# Iteramos para obtener la evolucion temporal de las bacterias

for i in range(len(t)):

    if (i%1000==0):

        print("Paso ",i, "de ", len(t))

    # Cada #cut datos los almacenamos en los arrays

    if i % cut == 0:

        x[k1] = ri[0]

        y[k1] = ri[1]

        X[k1] = Xi

        theta[k1] = thetai

        k1 += 1

        # Esperamos a que el sistema llegue a estado estacionario para calcular los campos

        if t[i] > 50:

            # Buscamos en que bin estan las bacterias y cuantas hay por intervalo

            # k es un vector con el bin de cada bacteria

            k = np.asarray(ri[0]/dx, dtype=int)

            # Obtenemos pares: indice (bins en que hay bacterias), cantidad de bacterias (por bin)

            count = Counter(k)

            # Extraemos los bins en que hay bacterias

            index = np.array(list(count.keys()))

            # Extraemos la cantidad de bacterias por bin en el instante

            rho_inst = np.array(list(count.values()))

            # Asignamos #(values) de bacterias en los bins #(index) del espacio discretizado

            rho_acum[index] = rho_acum[index] + rho_inst

            # Para calcular el resto de los campos, pasamos por cada bin en que hay alguna bacteria en el instante

            for q in index:

                # Identificamos cuales bacterias estan en el bin q

                bact_bin = np.where(k == q)

                # En el bin q, acumulamos las cantidades de interes de las bacterias

                # que se encuentran dentro de este

                rhox_inst = np.sum(Xi[bact_bin])

                Q1_inst = np.sum(np.cos(2*thetai[bact_bin]))

                Q2_inst = np.sum(np.sin(2*thetai[bact_bin]))

                JX1_inst = np.sum(np.cos(thetai[bact_bin])*Xi[bact_bin])

                JX2_inst = np.sum(np.sin(thetai[bact_bin])*Xi[bact_bin])

                Jx_inst = np.sum(V*np.cos(thetai[bact_bin]))

                Jy_inst = np.sum(V*np.sin(thetai[bact_bin]))

                # Sumamos a la densidad de los campos acumulada en los bins correspondientes

                rhox_acum[q] = rhox_acum[q] + rhox_inst

                Q1_acum[q] = Q1_acum[q] + Q1_inst

                Q2_acum[q] = Q2_acum[q] + Q2_inst

                JX1_acum[q] = JX1_acum[q] + JX1_inst

                JX2_acum[q] = JX2_acum[q] + JX2_inst

                Jx_acum[q] = Jx_acum[q] + Jx_inst

                Jy_acum[q] = Jy_acum[q] + Jy_inst

            #Contamos cuantas veces acumulamos los campos para luego promediar en el tiempo

            k2 += 1 

            # if k2%100 == 0:

            #     rho_mean = rho_acum/k2

            #     Jx_mean = Jx_acum/k2

            #     plt.plot(x_bin,Jx_mean)

            #     plt.xlabel('x')

            #     plt.ylabel(r'J$_x$')

            #     plt.tight_layout()

            #     # plt.savefig(r'Ratchet\Fields\J\Jx\c1 = {:04.1f} Tm = {:04.1f}.png'.format(c1,Tm),dpi=200)

            #     plt.show()

            #     plt.plot(x_bin,rho_acum)

            #     # plt.figtext(.75,.82,'$c_1$ = {:n}'.format(c1))

            #     # plt.figtext(.75,.77,r'$T_m=$ {:n}'.format(Tm))

            #     # plt.figtext(.75,.72,r'$N=${:n}'.format(N))

            #     # plt.figtext(.75,.67,r'$tsim=$ {:n}'.format(tsim))

            #     plt.xlabel('x')

            #     plt.ylabel(r'$\rho(x)$')

            #     plt.tight_layout()

            #     # plt.savefig(r'Ratchet\Fields\Bacterial density profile\c1 = {:04.1f} Tm = {:04.1f}.png'.format(c1,Tm),dpi=200)

            #     plt.show()





    # Probabilidad de tumbo para las N bacterias

    vi = v0*np.exp(alpha*Xi)

    # Posicion siguiente de las bacterias

    ri = ri + V*np.array([np.cos(thetai), np.sin(thetai)])*dt

    # Identificamos las bacterias que pasan a x < 0

    j1 = np.where(ri[0] <= 0)

    # Las bacterias que pasan a x < 0 vuelven a x + L

    ri[0][j1] = ri[0][j1] + L

    # En el caso contrario, si las bacterias llegan a x > L, pasan a x-L

    j2 = np.where(ri[0] >= L)

    ri[0][j2] = ri[0][j2] - L

    

    # Nuevo ligando

    ligandonuevo=ligando(ri[0])

    cdot=(ligandonuevo-ligandoprevio)/dt

    ligandoprevio=ligandonuevo

    

    # Concentracion de protei­na en instante siguiente

    Xi = Xi - dt*(Xi+B*cdot)/Tm + np.sqrt(2*dt/Tm)*rand.normal(0, 1, size=N)

    # Proceso de tumbo con probabilidad vi

    j3 = np.where(rand.uniform(0, 1, size=N) < vi*dt)

    thetai[j3] = rand.uniform(0, 2*np.pi, size=len(thetai[j3]))



# tiempo de ejecucion del codigo

end = time.perf_counter()

print(end-start)



# Avisa cuando termina de correr el codigo

# winsound.MessageBeep()



# Sacamos el promedio temporal de los campos

rho_mean = rho_acum/(k2*dx)



rhox_mean = rhox_acum/(k2*dx)

Q1_mean = Q1_acum/(k2*dx)

Q2_mean = Q2_acum/(k2*dx)

JXx_mean = JX1_acum/(k2*dx)
 

JXy_mean = JX2_acum/(k2*dx)      

Jx_mean = Jx_acum/(k2*dx)      
     

Jy_mean = Jy_acum/(k2*dx)      
     



# Perfil de densidad de bacterias

plt.plot(xx,rho_mean, marker="o",linestyle="")
plt.plot(xx,50+20*lorentzian_form(alpha,Tm,xx)*ligando(x_bin))

# plt.plot(x_bin,rho_inst, marker="+",linestyle="")


#! PARA GUARDAR DOCS EN .dat POR COLUMNAS

with open(os.getcwd()+'/density_lambda'+str(alpha)+'_.dat','a',newline='\n') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(zip(x_bin,rho_mean))
with open(os.getcwd()+'/densityx_lambda'+str(alpha)+'_.dat','a',newline='\n') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(zip(x_bin,rhox_mean))
with open(os.getcwd()+'/dipolarmom1_lambda'+str(alpha)+'_.dat','a',newline='\n') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(zip(x_bin,Q1_mean))
with open(os.getcwd()+'/dipolarmom2_lambda'+str(alpha)+'_.dat','a',newline='\n') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(zip(x_bin,Q2_mean))
with open(os.getcwd()+'/currentx_lambda'+str(alpha)+'_.dat','a',newline='\n') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(zip(x_bin,Jx_mean))
with open(os.getcwd()+'/currenty_lambda'+str(alpha)+'_.dat','a',newline='\n') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(zip(x_bin,Jy_mean))
with open(os.getcwd()+'/currentXx_lambda'+str(alpha)+'_.dat','a',newline='\n') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(zip(x_bin,JXx_mean))
with open(os.getcwd()+'/currentXy_lambda'+str(alpha)+'_.dat','a',newline='\n') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(zip(x_bin,JXy_mean))
# plt.figtext(.75,.82,'$c_1$ = {:n}'.format(c1)) # Sacar comentario si se quiere gráfico con leyenda

# plt.figtext(.75,.77,r'$T_m=$ {:n}'.format(Tm))

# plt.figtext(.75,.72,r'$N=${:n}'.format(N))

# plt.figtext(.75,.67,r'$tsim=$ {:n}'.format(tsim))

plt.xlabel('x')

plt.ylabel(r'$\rho(x)$')

plt.tight_layout()

# plt.savefig(r'Ratchet\Fields2\Bacterial density profile\c1 = {:04.1f} Tm = {:04.1f}.png'.format(c1,Tm),dpi=200)

plt.show()



# Perfil de densidad de bacterias semilog

log_rho = np.delete(rho_mean, np.where(rho_mean == 0))

log_xbin = np.delete(x_bin, np.where(rho_mean == 0))



plt.plot(log_xbin, log_rho)

plt.yscale('log')

# plt.figtext(.75, .82, '$c_1$ = {:n}'.format(c1))

# plt.figtext(.75, .77, r'$T_m=$ {:n}'.format(Tm))

# plt.figtext(.75, .72, r'$N=${:n}'.format(N))

# plt.figtext(.75, .67, r'$tsim=$ {:n}'.format(tsim))

plt.xlabel('x')

plt.ylabel(r'$\rho(x)$')

plt.tight_layout()

# plt.savefig(r'Ratchet\Fields2\Bacterial density profile\c1 = {:04.1f} Tm = {:04.1f} (semilog).png'.format(c1,Tm),dpi=200)

plt.show()



# Densidad de X

plt.plot(x_bin, rhox_mean)

# plt.figtext(.75, .82, '$c_1$ = {:n}'.format(c1))

# plt.figtext(.75, .77, r'$T_m=$ {:n}'.format(Tm))

# plt.figtext(.75, .72, r'$N=${:n}'.format(N))

# plt.figtext(.75, .67, r'$tsim=$ {:n}'.format(tsim))

plt.xlabel('x')

plt.ylabel(r'$\rho_X(x)$')

plt.tight_layout()

# plt.savefig(r'Ratchet\Fields2\X density\c1 = {:04.1f} Tm = {:04.1f}.png'.format(c1,Tm),dpi=200)

plt.show()



# Corriente de bacterias en x e y

plt.plot(x_bin,Jx_mean)

plt.xlabel('x')

plt.ylabel(r'J$_x$')

plt.tight_layout()

# plt.savefig(r'Ratchet\Fields2\J\Jx\c1 = {:04.1f} Tm = {:04.1f}.png'.format(c1,Tm),dpi=200)

plt.show()



plt.plot(x_bin,Jy_mean)

plt.xlabel('x')

plt.ylabel(r'J$_y$')

plt.tight_layout()

# plt.savefig(r'Ratchet\Fields2\J\Jy\c1 = {:04.1f} Tm = {:04.1f}.png'.format(c1,Tm),dpi=200)

plt.show()



#Campo dipolar Q1

plt.plot(x_bin,Q1_mean)

plt.xlabel('x')

plt.ylabel(r'$\langle$cos(2$\theta$)$\rangle$')

plt.tight_layout()

# plt.savefig(r'Ratchet\Fields2\Q1\c1 = {:04.1f} Tm = {:04.1f}.png'.format(c1,Tm),dpi=200)

plt.show()



#Campo dipolar Q2

plt.plot(x_bin,Q2_mean)

plt.xlabel('x')

plt.ylabel(r'$\langle$sin(2$\theta$)$\rangle$')

plt.tight_layout()

# plt.savefig(r'Ratchet\Fields2\Q2\c1 = {:04.1f} Tm = {:04.1f}.png'.format(c1,Tm),dpi=200)

plt.show()



#Corriente JXx

plt.plot(x_bin,JXx_mean)

plt.xlabel('x')

plt.ylabel(r'$\langle$X cos($\theta$)$\rangle$')

plt.tight_layout()

# plt.savefig(r'Ratchet\Fields2\JX\JX_x\c1 = {:04.1f} Tm = {:04.1f}.png'.format(c1,Tm),dpi=200)

plt.show()



#Corriente JXy

plt.plot(x_bin,JXy_mean)

plt.xlabel('x')

plt.ylabel(r'$\langle$X sin($\theta$)$\rangle$')

plt.tight_layout()

# plt.savefig(r'Ratchet\Fields2\JX\JX_y\c1 = {:04.1f} Tm = {:04.1f}.png'.format(c1,Tm),dpi=200)

plt.show()

















