import numpy as np
import scipy.optimize as opt             # para la funcion de gradiente
from matplotlib import pyplot as plt     # para dibujar las graficas
from valsLoader import *
from dataReader import load_mat

# g(X*Ot) = h(x)
def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def lambdaGraphic(errorX, errorXVal, lambdas):
    plt.figure()
    plt.plot(lambdas, errorX)
    plt.plot(lambdas, errorXVal)
    plt.show()

def coste(X, Y, O, reg):
    """devuelve la funcion de coste, dadas X, Y, y thetas"""
    O = O[np.newaxis]
    AuxO = O[:, 1:]

    H = sigmoid(np.dot(X, O.T))
    log1 = np.log(H).T
    Aux1 = np.dot(log1, Y)

    log2 = np.log(1 - H).T
    Aux2 = np.dot(log2, (1-Y))

    cost = (Aux1 + Aux2)/(len(X))
    
    return (-cost + (AuxO**2).sum()*reg/(2*X.shape[0]))[0, 0]

def gradiente(X, Y, O, reg):    
    """la operacion que hace el gradiente por dentro -> devuelve un vector de valores"""  
    AuxO = np.hstack([np.zeros([1]), O[1:,]])
    O = O[np.newaxis]
    AuxO = AuxO[np.newaxis].T
    
    return ((X.T.dot(sigmoid(np.dot(X, O.T))-Y))/X.shape[0] + (reg/X.shape[0])*AuxO)

def minimizeFunc(O, X, Y, reg):
    return (coste(X, Y, O, reg), gradiente(X, Y, O, reg))

def successPercentage(X, Y, O):
    """determina el porcentaje de aciertos comparando los resultados estimados con los resultados reales"""
    results = (sigmoid(X.dot(O[np.newaxis].T)) >= 0.5)
    results = (results == Y)
    return results.sum()/results.shape[0]

def main():
    X, Y, Xval, Yval, Xtest, Ytest = loadValues("steamReduced.csv")
    
    Xpoly = polynomize(X, 2)                   # pone automaticamente columna de 1s
    Xnorm, mu, sigma = normalize(Xpoly[:, 1:]) # se pasa sin la columna de 1s (evitar division entre 0)
    Xnorm = np.hstack([np.ones([Xnorm.shape[0], 1]), Xnorm]) # volvemos a poner columna de 1s

    XpolyVal = polynomize(Xval, 2)
    XnormVal = normalizeValues(XpolyVal[:, 1:], mu, sigma)
    XnormVal = np.hstack([np.ones([XnormVal.shape[0], 1]), XnormVal])

    XpolyTest = polynomize(Xtest, 2)
    XnormTest = normalizeValues(XpolyTest[:, 1:], mu, sigma)
    XnormTest = np.hstack([np.ones([XnormTest.shape[0], 1]), XnormTest])

    m = Xnorm.shape[0]      # numero de muestras de entrenamiento
    n = Xnorm.shape[1]      # numero de variables x que influyen en el resultado y, mas la columna de 1s

    thetaVec = np.zeros([n])

    l =  np.arange(0, 10, 0.25)

    errorX = np.zeros(l.shape[0])
    errorXVal = np.zeros(l.shape[0])

    # errores para cada valor de lambda
    for i in range(l.shape[0]):
        result = opt.minimize(fun = minimizeFunc, x0 = thetaVec,
         args = (Xnorm, Y, l[i]), method = 'TNC', jac = True, options = {'maxiter':700})
        O = result.x

        errorX[i] = coste(Xnorm, Y, O, l[i])
        errorXVal[i] = coste(XnormVal, Yval, O, l[i])

    lambdaGraphic(errorX, errorXVal, l)

    # lambda que hace el error minimo en los ejemplos de validacion
    lambdaIndex = np.argmin(errorXVal)
    print("Best lambda: " + str(l[lambdaIndex]))

    # thetas usando la lambda que hace el error minimo (sobre ejemplos de entrenamiento)
    result = opt.minimize(fun = minimizeFunc, x0 = thetaVec,
        args = (Xnorm, Y, l[lambdaIndex]), method = 'TNC', jac = True, options = {'maxiter':700})

    O = result.x

    #print(costeLineal(XnormTest, Ytest, O, l[lambdaIndex])) # error para los datos de testeo (nunca antes vistos)
    
    success = successPercentage(XnormTest, Ytest, O)
    
    print("Porcentaje de acierto: " + str(success*100) + "%")

main()