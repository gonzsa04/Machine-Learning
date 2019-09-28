import numpy as np
from pandas.io.parsers import read_csv
from matplotlib import pyplot as plt

def carga_csv(file_name):
    """carga el fichero csv especificado y lo devuelve en un array de numpy"""
    valores = read_csv(file_name, header=None).values

    return valores.astype(float)

def h(x, O):
    """devuelve la funcion h(x), bien usando la ecuacion de la recta o usando la matriz trasnpuesta de O"""
    kk = np.array(O)[np.newaxis]
    kk2 = np.array([1,x])[np.newaxis]
    #return  kk.T.dot(kk2)
    return O[0] + O[1]*x

def gradientDescendAlgorithm(valores):
    """minimiza la funcion de coste, hallando las O[0], O[1] que hacen el coste minimo, y por tanto, h(x) mas precisa"""
    O = [0,0]                    # O[0], O[1], inicialmente ambas a 0
    alpha = 0.01                 # coeficiente de aprendizaje
    m = np.size(valores, 0)      # numero de muestras de entrenamiento
    n = np.size(valores, 1) - 1  # numero de variables x que influyen en el resultado y

    # con 1500 iteraciones basta para encontrar O[0], O[1]
    for i in range(1500):
        sumatorio = [0,0]

        for rows in range(m):
            for cols in range(n):
                sumatorio[cols] += (h(valores[rows, cols], O) - valores[rows, cols])*valores[rows, cols]

        O[0] = O[0] - alpha*(1/m)*sumatorio[0]
        O[1] = O[1] - alpha*(1/m)*sumatorio[1]

    graphic(valores, O)  # pintamos la grafica. En un futuro esto tendra que estar en main, y este metodo devolver O[0], O[1] como una tupla

def graphic(valores, O):
    X = np.linspace(5, 22.5, 256, endpoint=True)
    Y = h(X, O)

    # pintamos muestras de entrenamiento
    plt.scatter(valores[:,1], valores[:,2], 0.3, 'red')
    # pintamos funcion de estimacion
    plt.plot(X, Y)

    plt.show()

def main():
    valores = carga_csv("ex1data1.csv")

    # añadimos una columna de 1s a la matriz de valores
    valores = np.hstack((np.ones(valores.shape[0])[np.newaxis].T, valores))
    
    # hallamos O[0], O[1] que minimicen el coste
    gradientDescendAlgorithm(valores)

    # pintamos la grafica

main()