import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

#### Constante de tolerancia
eps = 1.e-3; 

#Funciones
def f(x, y):
    return np.cos(3*x + 4*y) - np.cos(5*x - 2*y) 


#Parámetros del lattice
N = 50 # N0. de nodos 
x = np.linspace(0.0, 2.0*np.pi, N) # No. aleatorios en los arreglos
y = np.linspace(0.0, 2.0*np.pi, N)
Delta = 2.0*np.pi/len(x) # Longitud de los pasos
lat = np.random.rand(N, N)
lat_new = np.zeros((N, N), float)
F= np.zeros((N, N), float) #Genera un arreglo de NxN
for index, y_val in enumerate(y):
    valor = f(x, y_val)
    F[index] = valor

# Condiciones de fronetera:
cond_fron = 5.0
for i in range(0, N):
    lat[0, i] = cond_fron
    lat[N - 1, i] = cond_fron
    lat[i, 0] = cond_fron
    lat[i, N - 1] = cond_fron



count = 1
for i in range(N):
        for j in range(N):
            lat_new[i, j] = lat[i, j]   

# Implementación del algoritmo de Gauss-Seidel
while True:
    for i in range(1, len(x) - 1):
        for j in range(1, len(y) - 1):
            lat_new[i, j] = (lat[i+1, j] + lat_new[i-1, j] + lat[i, j+1] 
                             + lat_new[i, j-1] + F[i, j]*Delta)/4.0
    # Verifica la condición de tolerancia 
    if np.allclose(lat_new, lat, rtol=eps):
        break
    
    #Actualizamos los puntos del lattice
    for i in range(N):
        for j in range(N):
            lat[i, j] = lat_new[i, j]                
    
    count += 1
    print(f"Iteración: {count}")
    
# Graficación
x = list(range(0, N))
y = list(range(0, N))            

#Creamos la maya en el plano XY            
X , Y = plt.meshgrid(x, y)            

def altura(lat):
    z = lat[X, Y]
    return z

#Encontremos la altura en cada punto (X,Y)
Z = altura(lat_new)

fig = plt.figure(1)
ax = Axes3D(fig)
surf1 = ax.contour(X, Y, Z, N, cmap=cm.CMRmap) 
fig.colorbar(surf1, shrink=0.5, aspect=8)
ax.contour(X, Y, Z, zdir='z', offset=-0)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('$\phi(x,y)$')
plt.show()