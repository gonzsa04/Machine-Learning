import numpy as np
from pandas.io.parsers import read_csv
from matplotlib import pyplot as plt

def carga_csv(file_name):
    """carga el fichero csv especificado y lo devuelve en un array de numpy"""
    valores = read_csv(file_name, header=None).values

    return valores.astype(float)

def hTransposed(X, O):
    """devuelve la funcion h(x) usando la matriz trasnpuesta de O"""
    return O[0]*X[0] + O[1]*X[1]

def hLine(X, O):
    """devuelve la funcion h(x) usando la ecuacion de la recta"""
    return O[0] + O[1]*X

def gradientDescendAlgorithm(valores):
    """minimiza la funcion de coste, hallando las O[0], O[1] que hacen el coste minimo, y por tanto, h(x) mas precisa"""
    m = np.size(valores, 0)      # numero de muestras de entrenamiento
    n = np.size(valores, 1) - 1  # numero de variables x que influyen en el resultado y
    O = np.zeros(n)              # O[0], O[1], inicialmente ambas a 0
    alpha = 0.01                 # coeficiente de aprendizaje

    # con 1500 iteraciones basta para encontrar O[0], O[1]
    for i in range(1500):
        sumaJ = 0
        sumatorio = np.zeros(n)

        for rows in range(m):
            for cols in range(n):
                sumatorio[cols] += (hLine(valores[rows, cols], O) - valores[rows, cols])*valores[rows, cols]
                sumaJ += ((hLine(valores[rows, cols], O) - valores[rows, cols]))**2
        
        """for rows in range(m):
            sumatorio = [0,0]
            h = hTransposed(valores[rows, 0:-1], O)
            #[start_row_index : end_row_index , start_column_index : end_column_index] para coger partes de una matriz
            for cols in range(n):
                sumatorio[cols] += (h - valores[rows, cols])*valores[rows, cols]
                sumaJ += ((hLine(valores[rows, cols], O) - valores[rows, cols]))**2"""

        print((1/(2*m))*sumaJ)
        O = O - alpha*(1/m)*sumatorio

    graphic(valores, O)  # pintamos la grafica. En un futuro esto tendra que estar en main, y este metodo devolver O[0], O[1] como una tupla

def graphic(valores, O):
    X = np.linspace(5, 22.5, 256, endpoint=True)
    Y = hLine(X, O)

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