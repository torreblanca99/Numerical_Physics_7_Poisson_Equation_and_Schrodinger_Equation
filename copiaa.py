"""
@author: Julio C. Torreblanca
"""

import numpy as np
import matplotlib.pyplot as plt

################ Constantes
eps = 1e-4; Nsteps = 501; h = 0.04; Nmax = 100 #Parámetros
E = -50.0; Emax = 1.1*E; Emin = E/1.1


###############Constante 2m/(hc)^2
m = 0.511 #masa del electrón [MeV /c^2]
hc = 0.197 #0.4829 Planck*c [MeV * pm]
Cte = 2.*m/(hc*hc)
###############Constantes del pozo de potencial
a = 10. #Longitud del pozo en [pm]
Vx = -60. #Potencial en [Mev]


##############Variables globales


############### Funciones

    
    
def f(x, y):
    """Esta función calcula las funciones f^(0) y f^(1) de la forma estándar
    en la solución de la ODE.
    
    Parámetros: 
        x: punto a ser evaluado1
        y: arreglo que contiene los valores de y^(0) y y^(1) de a forma estándar
            en la solición de la EDO
    
    Salida:
        f: arregloq ue contiene las funciones f^(0) y f^(1) evaluadas 
    """
    #global E
    F = np.zeros(2,float)
    F[0] = y[1]               #Forma estándar f^(0)
    F[1] = -(Cte)*(E-V(x))*y[0] #Forma estándar f^(1)
    return F

def V(x: float)->float:
    """Esta función calcula el potencial cumpliendo con la condición 
    en el infinito
    
    Parámetros:
        x: punto donde se quiere obtener el potencial
    Salida:
        Valor del potencial
    """
    
    if abs(x) <= a :    return Vx
    else:   return 0.



def rk4Algor(x,h,N,y,f): 
    """Esta función resulve la ODE por runge kutta orden 4
    
    Parámetros:
        x: puntos donde se evaluará
        h: longitud del salto en el cálculo de la integral
        N: longitud de los arreglos para el cálculo en varias variables
        y: variable donde se guardaran los resultados
        f: funcion que calcula las formas estándares f^(0) y f^(1) en el algoritmo
        
    Salida:
        y: un arreglo que contiene todos los puntos evaluados en la solución de la EDO
    
    """
    k1=np.zeros(N); k2=np.zeros(N); k3=np.zeros(N); k4=np.zeros(N)
    k1 = h*f(x,y)                             
    k2 = h*f(x+h/2.,y+k1/2.)
    k3 = h*f(x+h/2.,y+k2/2.)
    k4 = h*f(x+h,y+k3)
    y=y+(k1+2*(k2+k3)+k4)/6.
    return y  


def diff(h, E):
    """Esta función calcula la solución para la función de onda por izquierda 
    y por derecha y las conecta en el punto de pegado. Además evalúa la 
    condición de continuidad para la función y su derivada.
    
    Paránmetros:
        h: longitud de los pasos en el cálculo de la derivada
        E: Valor de la energía
    Salida:
        Regresa de Delta para la condición de continuidad de la función 
        de onda y su deriavda
    
    """
    y = np.zeros(2, float)
    i_match = Nsteps//3             #Encajando el radio
    nL = i_match + 1
    y[0] = 1.E-15                     #Calculo de la función exponencial en infinito
    y[1] = y[0]*np.sqrt(-E*Cte)  #Calculo de la derivada de la exponencial
    for ix in range(0, nL+1):
        x = h * (ix - Nsteps/2)
        y = rk4Algor(x, h, 2, y, f)
    left = y[1]/y[0]                #Derivada logarítmica
    y[0] = 1.E-15                   #Inicializa R wf
    y[1] = -y[0] * np.sqrt(-E*Cte)#Derivada de la onda derecha en reversa
    for ix in range (Nsteps, nL + 1, -1):
        x = h*(ix + 1 - Nsteps/2)
        y = rk4Algor(x, -h, 2, y, f)
    right = y[1]/y[0]               #Derivadas logaritmicas
    return ((left - right)/(left + right))

def plot(h):
    global xL, xR, Rwf, Lwf
    
    x = 0.      #Radio de pegado  
    Lwf = []    #Función de onda por la izquierda
    Rwf = []    #Función de onda por la derecha
    xR = []     #x para Rwf
    xL = []     #x para Lwf
    Nsteps = 1501       #No. de pasos para la integración de la ODE
    y = np.zeros(2, float)
    yL = np.zeros ((2,505), float)
    i_match = 500   #Radio de pegado
    nL = i_match + 1
    #print('nL ',nL)
    y[0] = 1.E-40      #Función de onda izquierda inicial
    y[1] = -np.sqrt(-E*Cte) * y[0] #Derivada valuada en x<0
    for ix in range(0, nL + 1):
        yL[0][ix] = y[0]
        yL[1][ix] = y[1]
        x = h * (ix - Nsteps/2)
        y = rk4Algor(x, h, 2, y, f)
    y[0] = -1.E-15 
    y[1] = -np.sqrt(-E*Cte)*y[0]
    for ix in range(Nsteps - 1, nL + 2, -1):    #Función de onda derecha
        x = h * (ix + 1 - Nsteps/2)     #Integración
        y = rk4Algor(x, -h, 2, y, f)
        xR.append(x)
        Rwf.append(y[0])
    x = x - h
    normL = y[0]/yL[0][nL]
    for ix in range(0,nL + 1):  #Normaliza la función de onda izquierda y la deriva
        x = h*(ix - Nsteps/2 +1)
        y[0] = yL[0][ix]*normL
        y[1] = yL[1][ix]*normL
        xL.append(x)
        Lwf.append(y[0])    #Factor de escala
    
fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid()       #j +=1
    
for count in range (0, Nmax):           #Main program 
    E = (Emax + Emin)/2                 #Rango de bisección
    Diff = diff(h, E)
    Etemp = E
    E = Emax
    diffMax = diff(h,E)
    E = Etemp
    if (diffMax * Diff > 0):    Emax = E    #Algoritmo de bisección
    else:   Emin = E
    print(f'Iteración {count}, E = {E}')
    if (abs(Diff) < eps ):  break
    if count > 3:
        fig.clear()
        plot(h)
        plt.plot(xL, Lwf)
        plt.plot(xR,Rwf)
        plt.text(3, -200, f'Energía = {E:.4f}', fontsize = 14)
        plt.pause(0.8)  #Pausa entre figuras
    
    plt.xlabel('x')
    plt.ylabel('$\psi(x) $', fontsize = 18)
    plt.title('Funciones de onda Izq. y Dcha. pegadas en x = 0')

print(f'Eigenvalor final E = {E}')
print(f'Iteraciones = {count}, max = {Nmax}')
plt.show()

