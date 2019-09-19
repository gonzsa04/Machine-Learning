import numpy as np
import scipy.integrate
from matplotlib import pyplot as plt

# integral de una funcion usando arrays de numpy
def integra_mc_npy(fun = np.sin, a = 0, b = 3, num_puntos = 10000):
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
    

    plt.savefig('IntegralMonteCarlo.png')
    plt.show()

def integra_mc_list(fun = np.sin, a = 0, b = 3, num_puntos = 10000):
    X = np.linspace(a, b, 256, endpoint=True)
    Y = fun(X)

    maxValue = np.amax(Y)

    xRand = np.random.uniform(a, b, num_puntos)
    yRand = np.random.uniform(0, maxValue, num_puntos)
    nDebajo = (fun(xRand) > yRand)

    IMonteCarlo = (nDebajo.sum()/num_puntos)*(b-a)*maxValue
    IIntegrate = (scipy.integrate.quad(fun, a, b))[0]


    plt.axes([0, 0, 1, 1])
    plt.text(0.05,0.05, "Integral por Monte Carlo = " + str(IMonteCarlo))
    plt.text(0.05,0.1, "Integral por scipy = " + str(IIntegrate))

    plt.axes([.07, .2, .9, .75])
    plt.plot(X, Y)
    plt.scatter(xRand, yRand, 0.1, 'red')
    
    plt.show()

def main():
    integra_mc_npy()
    integra_mc_list()

main()