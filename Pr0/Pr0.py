import numpy as np
import scipy.integrate
from matplotlib import pyplot as plt
import random
import time

# integral de una funcion usando arrays de numpy
def integra_mc_npy(fun = np.sin, a = 0, b = 3, num_puntos = 10000):
    tic = time.proccess_time() # tiempo inicial

    # generacion de la curva de la funcion
    X = np.linspace(a, b, 256, endpoint=True)
    Y = fun(X)

    maxValue = np.amax(Y) # maximo valor de la funcion

    #generacion de numeros aleatorios
    xRand = np.random.uniform(a, b, num_puntos)
    yRand = np.random.uniform(0, maxValue, num_puntos)

    nDebajo = (fun(xRand) > yRand) # los que quedan bajo la funcion

    # integrales por monte carlo y scipy
    IMonteCarlo = (nDebajo.sum()/num_puntos)*(b-a)*maxValue
    IIntegrate = (scipy.integrate.quad(fun, a, b))[0]

    #pintamos grafica y datos
    plt.axes([0, 0, 1, 1])
    plt.text(0.05,0.05, "Integral por Monte Carlo = " + str(IMonteCarlo))
    plt.text(0.05,0.1, "Integral por scipy = " + str(IIntegrate))

    plt.axes([.07, .2, .9, .75])
    plt.plot(X, Y)
    plt.scatter(xRand, yRand, 0.1, 'red')

    plt.title("HECHO CON ARRAYS DE NUMPY")

    plt.savefig('IntegralMonteCarloNumPY.png')
    plt.show()

    toc = time.proccess_time() # tiempo final
    return 1000 * (toc - tic)



def integra_mc_list(fun = np.sin, a = 0, b = 3, num_puntos = 10000):
    tic = time.proccess_time() # tiempo inicial

    step = 256
    espaciado = (b - a)/step
    X = []
    Y = []
    maxValue = -float("inf")
    
    # generacion de la curva de la funcion
    for i in range(step):
        X.append(i * espaciado)
        Y.append(fun(X[i]))
        if(Y[i] > maxValue):
            maxValue = Y[i] # maximo valor de la funcion

    xRand = []
    yRand = []
    nDebajo = 0

    #generacion de numeros aleatorios
    for i in range(num_puntos):
        xRand.append(random.uniform(a, b))
        yRand.append(random.uniform(0, maxValue))
        if(fun(xRand[i]) > yRand[i]):
             nDebajo = nDebajo + 1  # los que quedan bajo la funcion

    # integrales por monte carlo y scipy
    IMonteCarlo = (nDebajo/num_puntos)*(b-a)*maxValue
    IIntegrate = (scipy.integrate.quad(fun, a, b))[0]

    #pintamos grafica y datos
    plt.axes([0, 0, 1, 1])
    plt.text(0.05,0.05, "Integral por Monte Carlo = " + str(IMonteCarlo))
    plt.text(0.05,0.1, "Integral por scipy = " + str(IIntegrate))

    plt.axes([.07, .2, .9, .75])
    plt.plot(X, Y)
    plt.scatter(xRand, yRand, 0.1, 'red')

    plt.title("HECHO CON LISTAS DE PYTHON")
    
    plt.savefig('IntegralMonteCarloLists.png')
    plt.show()

    toc = time.proccess_time() # tiempo final
    return 1000 * (toc - tic)



def compara_tiempos():
    X = np.linspace(100, 10000000, 20)
    timeNumPY = []
    timeList = []

    for x in X:
        numPuntos = np.random.uniform()



def main():
    timeNumPY = integra_mc_npy()
    timeList = integra_mc_list()

main()