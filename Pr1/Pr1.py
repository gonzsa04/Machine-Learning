import numpy as np
from pandas.io.parsers import read_csv
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def carga_csv(file_name):
    """carga el fichero csv especificado y lo devuelve en un array de numpy"""
    valores = read_csv(file_name, header=None).values

    return valores.astype(float)

def hTransposed(X, O):
    """devuelve la funcion h(x) usando la matriz transpuesta de O"""
    AuxO = O[np.newaxis]
    AuxX = X[np.newaxis]
    return (np.dot(np.transpose(AuxO), AuxX)).sum()

def hLine(X, O):
    """devuelve la funcion h(x) usando la ecuacion de la recta"""
    return O[0] + O[1]*X

def coste(X, Y, O):
    H = np.dot(X, O)
    Aux = (H-Y)**2
    return Aux.sum()/(2*len(X)) # lo mismo que hacer la formula con el sumatorio...

def gradientDescendAlgorithm(X, Y, alpha, m, n):
    """minimiza la funcion de coste, hallando las O[0], O[1] que hacen el coste minimo, y por tanto, h(x) mas precisa"""
   
    O = np.zeros(n)      # O[0], O[1], inicialmente ambas a 0

    # con 1500 iteraciones basta para encontrar O[0], O[1]
    for i in range(1500):
        sumatorio = np.zeros(n)

        for rows in range(m):      # bucle utilizando la ecuacion de la recta para h (obviamos la columna de 1s)
            for cols in range(n):
                sumatorio[cols] += (hLine(X[rows, 1], O) - Y[rows])*X[rows, cols]
        
        """for rows in range(m):
            h = hTransposed(X[rows], O)
            for cols in range(n):
                sumatorio[cols] += (h - Y[rows])*X[rows, cols]"""

        #print(coste(X, Y, O))
        O = O - alpha*(1/m)*sumatorio

    return O

def functionGraphic(X, Y, O):
    x = np.linspace(5, 22.5, 256, endpoint=True)
    y = hLine(x, O)

    # pintamos muestras de entrenamiento
    plt.scatter(X[:, 1], Y, 1, 'red')
    # pintamos funcion de estimacion
    plt.plot(x, y)

    plt.show()

def costGraphics(X, Y):
    # grafica 3D
    fig = plt.figure()
    ax = fig.gca(projection = '3d')

    Theta0, Theta1, Coste = make_data([-10, 10], [-1, 4], X, Y)

    ax.plot_surface(Theta0, Theta1, Coste, cmap = cm.Spectral, linewidth = 0, antialiased = False)
    plt.show()

    # contour
    plt.contour(Theta0, Theta1, Coste, colors = 'red')
    ax.contour(Theta0, Theta1, Coste, np.logspace(-2, 3, 20)) # lista con los ticks de las curvas de nivel
                                                              # con escala logaritmica de 20 valores entre 10^-2 y 10^-3
    plt.show()



def make_data(t0_range, t1_range, X, Y):
    """Genera las  matrices X, Y, Z para generar un plot en 3D"""

    step = 0.1
    Theta0 = np.arange(t0_range[0], t0_range[1], step)
    Theta1 = np.arange(t1_range[0], t1_range[1], step)

    Theta0, Theta1 = np.meshgrid(Theta0, Theta1)
    #Theta0 y Theta1 tienen las mismas dimensiones, de forma que cogiendo un elemento de cada uno se generan las coordenadas x, y
    # de todos los puntos de la rejilla

    Coste = np.empty_like(Theta0)

    for ix, iy in np.ndindex(Theta0.shape):
        Coste[ix, iy] = coste(X, Y, [Theta0[ix, iy], Theta1[ix, iy]])

    return [Theta0, Theta1, Coste]

def main():
    valores = carga_csv("ex1data1.csv")

    X = valores[:, :-1] # matriz X, con todas las filas y todas las columnas menos la ultima (ys)
    Y = valores[:, -1]  # matriz Y, con todas las filas y la ultima columna
    #print(X.shape)
    #print(Y.shape)

    m = X.shape[0]      # numero de muestras de entrenamiento

    # Se colocan m filas de 1 columna de 1s al principio (concatenacion de matrices)
    X = np.hstack([np.ones([m, 1]), X])

    n = X.shape[1]      # numero de variables x que influyen en el resultado y
    alpha = 0.01        # coeficiente de aprendizaje

    # hallamos O[0], O[1] que minimicen el coste
    O = gradientDescendAlgorithm(X, Y, alpha, m, n)

    # pintamos graficas
    if n < 3:                     # solo lo pintaremos si no tiene mas de dos variables x (pasaria a ser multidimensional)
        functionGraphic(X, Y, O)  # pintamos la grafica. En un futuro esto tendra que estar en main, y este metodo devolver O[0], O[1] como una tupla
    costGraphics(X, Y)             # graficos para ver el coste

main()