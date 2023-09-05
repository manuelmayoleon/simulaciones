import time
import os
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rand
from collections import Counter
# from matplotlib import colors
import matplotlib_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')


# Paso de tiempo delta t
dt = .0001
# Cantidad de bacterias que simulamos
N = 2000
# Parametros del sistema
c1 = 1
Tm = 1
v0 = V = B = D = alpha = 1
# Tiempo hasta el que se simula
tsim = 100
# Intervalo de tiempo de la simulacion
t = np.arange(0, tsim, dt)
# Cada cuantos pasos dt almacenamos los datos
cut = 1000
# Dimension de la simulacion
d = 2

# Largo caracteri�stico
L = d/(2*alpha*c1)
# Tiempo caracteristico
tau = L**2/D

# Hasta donde mediremos densidad, de bacterias, \rho(y)
# si el dy es muy chico e ysup es grande, los resultados salen con mucho ruido.
# el definir ysup de esta manera funciona bastante bien para Tm = 1, no logre encontrar una
# buena dependencia en el tiempo de memoria asi que para Tm = 10 o Tm = 0.1 hay que ir probando el ysup bueno
ysup = 20*L

# Discretizacion del espacio
dy = ysup/300
yfield = np.arange(0,ysup,dy)
y_bin = yfield+0.5*dy

# Condiciones iniciales para las bacterias, exponencial en y, y todas en x = 0
yi = rand.exponential(L, size=N)
xi = np.zeros(N)
# Posicion bacterias
ri = np.array([xi, yi])

# Concentracion inicial de protei�na en las bacterias
Xi = rand.normal(0, 1, size=N)
# Orientacion inicial de las bacterias
thetai = rand.uniform(0, 2*np.pi, size=N)

# Arreglos para almacenar los datos
x = np.zeros([int(len(t)/cut), N])
y = np.zeros([int(len(t)/cut), N])
X = np.zeros([int(len(t)/cut), N])
theta = np.zeros([int(len(t)/cut), N])

# Contador para almacenar los datos
k1 = 0
# Contador para densidad
k2 = 0

# Arreglos para almacenar la densidades acumuladas
rho_acum = np.zeros([len(yfield)])

rhox_acum = np.zeros(len(yfield))
Q1_acum = np.zeros([len(yfield)])
Q2_acum = np.zeros([len(yfield)])
JX1_acum = np.zeros([len(yfield)])
JX2_acum = np.zeros([len(yfield)])
f12_acum = np.zeros([len(yfield)])
Jx_acum = np.zeros([len(yfield)])
Jy_acum = np.zeros([len(yfield)])


# Para ver tiempo de ejecucion
start = time.perf_counter()

# Iteramos para obtener la evolucion temporal de las bacterias
for i in range(len(t)):
    # Cada #cut datos los almacenamos en los arrays
    if i % cut == 0:
        x[k1] = ri[0]
        y[k1] = ri[1]
        X[k1] = Xi
        theta[k1] = thetai
        k1 += 1
        # Esperamos a que el sistema llegue a estado estacionario para obtener perfiles de densidad
        if t[i] > max(5*tau, 5*Tm):
            # Buscamos en que bin estan las bacterias y cuantas hay por intervalo
            # k es un vector con el bin de cada bacteria
            k = np.asarray(ri[1]/dy, dtype=int)
            # Las bacterias que quedan fuera del rango de yfield las olvidamos para evitar problemas con los indices
            k = np.delete(k, np.where(k >= len(yfield)))
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
                ########### Revisar normalizacion de los campos ##########
                # Identificamos cuales bacterias estan en el bin q
                bact_bin = np.where(k == q)
                # En el bin q, promediamos las cantidades de interes de las bacterias
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
    # Probabilidad de tumbo para las N bacterias
    vi = v0*np.exp(alpha*Xi)
    # Posicion siguiente de las bacterias
    ri = ri + V*np.array([np.cos(thetai), np.sin(thetai)])*dt
    # Si y<0 para alguna bacteria, se asigna a y=0 (se queda en la pared)
    j1 = np.where(ri[1] <= 0)
    ri[1][j1] = 0
    # Derivada de comida para cada bacteria
    cdot = -V*c1*(np.sin(thetai))
    # Para las bacterias en la pared, el gradiente es 0
    cdot[j1] = 0
    # Concentracion de protei�na en instante siguiente
    Xi = Xi - dt*(Xi+B*cdot)/Tm + np.sqrt(2*dt/Tm)*rand.normal(0, 1, size=N)
    # Proceso de tumbo con probabilidad vi
    j2 = np.where(rand.uniform(0, 1, size=N) < vi*dt)
    thetai[j2] = rand.uniform(0, 2*np.pi, size=len(thetai[j2]))

# tiempo de ejecucion del codigo
end = time.perf_counter()
print(end-start)

# Sacamos el promedio temporal de los campos
rho_mean = rho_acum/k2
rhox_mean = rhox_acum/k2
Q1_mean = Q1_acum/k2
Q2_mean = Q2_acum/k2
JXx_mean = JX1_acum/k2
JXy_mean = JX2_acum/k2
Jx_mean = Jx_acum/k2
Jy_mean = Jy_acum/k2

####################### Plots #############################

# # Perfil de densidad de bacterias
# plt.plot(y_bin,rho_mean)
# plt.title('Bacterial density')
# plt.figtext(.75,.82,'$c_1$ = {:n}'.format(c1))
# plt.figtext(.75,.77,r'$T_m=$ {:n}'.format(Tm))
# plt.figtext(.75,.72,r'$N=${:n}'.format(N))
# plt.figtext(.75,.67,r'$tsim=$ {:n}'.format(tsim))
# plt.xlabel('y')
# plt.ylabel(r'$\rho(y)$')
# plt.savefig('Pared\Fields\Bacterial density profile\c1 = {:04.1f} Tm = {:04.1f}.png'.format(c1,Tm),dpi=200)
# plt.show()

# # Perfil de densidad de bacterias en semilog
# log_rho = np.delete(rho_mean, np.where(rho_mean == 0))
# log_ybin = np.delete(y_bin, np.where(rho_mean == 0))

# plt.plot(log_ybin, log_rho)
# plt.yscale('log')
# plt.title('Bacterial density')
# # plt.figtext(.75, .82, '$c_1$ = {:n}'.format(c1))
# # plt.figtext(.75, .77, r'$T_m=$ {:n}'.format(Tm))
# # plt.figtext(.75, .72, r'$N=${:n}'.format(N))
# # plt.figtext(.75, .67, r'$tsim=$ {:n}'.format(tsim))
# plt.xlabel('y')
# plt.ylabel(r'$\rho(y)$')
# plt.savefig('Pared\Fields\Bacterial density profile\c1 = {:04.1f} Tm = {:04.1f} (semilog).png'.format(c1,Tm),dpi=200)
# plt.show()

# # Densidad de X
# plt.plot(y_bin, rhox_mean)
# plt.title('X density')
# # plt.figtext(.75, .82, '$c_1$ = {:n}'.format(c1))
# # plt.figtext(.75, .77, r'$T_m=$ {:n}'.format(Tm))
# # plt.figtext(.75, .72, r'$N=${:n}'.format(N))
# # plt.figtext(.75, .67, r'$tsim=$ {:n}'.format(tsim))
# plt.xlabel('y')
# plt.ylabel(r'$\rho_X(y)$')
# plt.savefig('Pared\Fields\X density profile\c1 = {:04.1f} Tm = {:04.1f}.png'.format(c1,Tm),dpi=200)
# plt.show()

# # Corriente en x e y
# plt.plot(y_bin,Jx_mean)
# plt.xlabel('y')
# plt.ylabel(r'J$_x$')
# plt.savefig(r'Pared\Fields\J\Jx\c1 = {:04.1f} Tm = {:04.1f}.png'.format(c1,Tm),dpi=200)
# plt.show()

# plt.plot(y_bin,Jy_mean)
# plt.xlabel('y')
# plt.ylabel(r'J$_y$')
# plt.savefig(r'Pared\Fields\J\Jy\c1 = {:04.1f} Tm = {:04.1f}.png'.format(c1,Tm),dpi=200)
# plt.show()

# #Campo dipolar Q1
# plt.plot(yfield,Q1_mean)
# plt.xlabel('y')
# plt.ylabel(r'$\langle$cos(2$\theta$)$\rangle$')
# plt.savefig('Pared\Fields\Q1\c1 = {:04.1f} Tm = {:04.1f}.png'.format(c1,Tm),dpi=200)
# plt.show()

# #Campo dipolar Q2
# plt.plot(yfield,Q2_mean)
# plt.xlabel('y')
# plt.ylabel(r'$\langle$sin(2$\theta$)$\rangle$')
# plt.savefig('Pared\Fields\Q2\c1 = {:04.1f} Tm = {:04.1f}.png'.format(c1,Tm),dpi=200)
# plt.show()

# #Corriente JXx
# plt.plot(yfield,JXx_mean)
# plt.xlabel('y')
# plt.ylabel(r'$\langle$X cos($\theta$)$\rangle$')
# plt.savefig('Pared\Fields\JX\JXx\c1 = {:04.1f} Tm = {:04.1f}.png'.format(c1,Tm),dpi=200)
# plt.show()

# #Corriente JXy
# plt.plot(yfield,JXy_mean)
# plt.xlabel('y')
# plt.ylabel(r'$\langle$X sin($\theta$)$\rangle$')
# plt.savefig('Pared\Fields\JX\JXy\c1 = {:04.1f} Tm = {:04.1f}.png'.format(c1,Tm),dpi=200)
# plt.show()


