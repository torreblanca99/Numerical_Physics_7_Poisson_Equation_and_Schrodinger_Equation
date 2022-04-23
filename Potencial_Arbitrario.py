import numpy as np
import matplotlib.pyplot as plt

################ Constantes
eps = 1e-2; Nsteps = 501; h = 0.04; Nmax = 100 #Parámetros
E = -17.0; Emax = 1.1*E; Emin = E/1.1


###############Constante 2m/(hc)^2
m = 0.511 #masa del electrón [MeV /c^2]
hc = 0.197 #0.4829 Planck*c [MeV * pm]
#Cte = 2.*m/(hc*hc)
Cte = 0.4829 
###############Constantes del pozo de potencial
a = 10. #Longitud del pozo en [pm]
Vx = -20. #Potencial en [Mev]
Xinf = a*3#Infinito -> no hacer muy grande porque si no tocamos overflow

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
    kappa = np.sqrt(-E*Cte) 
    f_onda = 1/np.exp(kappa*Xinf)  
    
    #Cálculo por la izquierda
    y = np.zeros(2, float)      #Arreglo que contendrá la función de onda y su derivada
    x_m = Nsteps//3            #Este es para dividir las tres partes y encjar en x=-a
    nL = x_m + 1            #Con esta parte lograremos encajar en x=-a
    y[0] = f_onda       #Inicialización de la función de onda izq en infinito
    y[1] = kappa*y[0]   #Derivada de la función de onda izq en infinito
    for ix in range(0, nL+1):  #Va de -X_inf hasta el punto de pegado
        x = h * (ix - Nsteps/2)
        y = rk4Algor(x, h, 2, y, f)
    izq= y[1]/y[0]        #Derivada logarítmica por izq
    
    #Cálculo por la derecha
    y[0] = f_onda         #Inicializa la función de onda dcha en infinito
    y[1] = -kappa*y[0]    #Derivada de la fun. de onda derecha en infinito
    for ix in range (Nsteps, nL + 1, -1):  #Va de X_inf hasta el x_m en reversa
        x = h*(ix + 1 - Nsteps/2)
        y = rk4Algor(x, -h, 2, y, f)    #Integración en reversa
    dcha = y[1]/y[0]               #Derivadas logaritmicas
    
    
    return ((izq- dcha)/(izq + dcha))

def plot(h,E):
    """Esta función realiza el cálculo de la fun. de onda por izquierda y 
    por derecha con las condiciones de frontera y las normaliza.
    
    Parámetros:
        h: longitud de los pasos para la integración
        E: energía para el cálculo de la conste
    Salidas:
        xL: arreglo con los puntos x para la fun de onda izq
        Lwf: arreglo con los puntos de la fun de onda izq valuada en cada punto
            de xL
        xR: arreglo con los puntos x para la fun de onda dcha
        Rwf: arreglo con los puntos de la fun de onda dcha valuada en cada punto
            de xR
    """
    Lwf = []    #Función de onda por la izquierda
    Rwf = []    #Función de onda por la derecha
    xR = []     #x para Rwf
    xL = []     #x para Lwf
    Nsteps = 1501       #No. de pasos para la integración de la ODE
    y = np.zeros(2, float) #Arreglo apr ala fun de onda y su derivada
    yL = np.zeros ((2,505), float) #Arreglo auxiliar para calcular las fun de onda
    i_match = 500   #Paso de pegado
    nL = i_match + 1
    
    #Cálculo de la fun. de onda en el infinito
    kappa = np.sqrt(-E*Cte) 
    f_onda = 1/np.exp(kappa*Xinf)  
    
    
    #Cálculo de la fun de onda por la izquierda
    y[0] = f_onda      #Función de onda izquierda inicial
    y[1] = kappa*y[0]  #Derivada valuada en x<0
    
    for ix in range(0, nL + 1):
        yL[0][ix] = y[0]
        yL[1][ix] = y[1]
        x = h * (ix - Nsteps/2)
        y = rk4Algor(x, h, 2, y, f)
    
    #Cálculo función de onda po derecha    
    y[0] = f_onda 
    y[1] = kappa*y[0]
    for ix in range(Nsteps - 1, nL + 2, -1):    #Función de onda derecha
        x = h * (ix + 1 - Nsteps/2)     #Integración
        y = rk4Algor(x, -h, 2, y, f)
        xR.append(x)
        Rwf.append(y[0])

    
    normL = y[0]/yL[0][nL]
    for ix in range(0,nL + 1):  #Normaliza la función de onda izquierda y la deriva
        x = h*(ix - Nsteps/2 +1)
        y[0] = yL[0][ix]*normL
        y[1] = yL[1][ix]*normL
        xL.append(x)
        Lwf.append(y[0])    
    return xL,Lwf,xR,Rwf




fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid()       


####Parte principal    
for i in range (0, Nmax):     
    E = (Emax + Emin)/2                 #Rango de bisección
    #Algoritmo de bisección
    Diff = diff(h, E)
    Eaux = E
    E = Emax
    diffMax = diff(h,E)
    E = Eaux
    if (diffMax * Diff > 0):    
        Emax = E    
    else:   
        Emin = E
    print(f'Iteración {i}, E = {E:.4f}')
      
    fig.clear()
    x_izq, onda_izq, x_dcha, onda_dcha = plot(h,E)
    plt.plot(x_izq, onda_izq)
    plt.plot(x_dcha,onda_dcha)
    plt.pause(0.8)  #Pausa entre figuras
    
    if (abs(Diff) < eps ):  break


plt.xlabel('x')
plt.ylabel('$\psi(x) $', fontsize = 18)
plt.title('Funciones de onda Izq. y Dcha. pegadas en x = -a')

print(f'Eigenvalor final E = {E:.4f}')
print(f'Iteraciones = {i}, max = {Nmax}')
plt.show()
