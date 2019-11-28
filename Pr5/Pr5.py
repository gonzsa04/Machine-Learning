import numpy as np
import scipy.optimize as opt             # para la funcion de gradiente
from matplotlib import pyplot as plt     # para dibujar las graficas
from scipy.io import loadmat

def load_mat(file_name):
    """carga el fichero mat especificado y lo devuelve en una matriz data"""
    return loadmat(file_name)

def hLine(X, O):
    """devuelve la funcion h(x) usando la ecuacion de la recta"""
    return O[0] + O[1]*X

def functionGraphic(X, errorX, errorXVal):
    """muestra el grafico de la funcion h(x)"""

    x = np.linspace(0, 11, errorX.shape[0], endpoint=True)
    xVal = np.linspace(0, 11, errorX.shape[0], endpoint=True)

    # pintamos funcion de estimacion
    plt.plot(x, errorX)
    plt.plot(xVal, errorXVal)
    plt.savefig('LearningCurve.png')
    plt.show()

def costeLineal(X, Y, O, reg):
    """devuelve la funcion de coste, dadas X, Y, y thetas"""
    AuxO = O[1:]
    O = O[np.newaxis]

    H = np.dot(X, O.T)
    Aux = (H-Y)**2
    cost = Aux.sum()/(2*len(X))

    return cost + (AuxO**2)*reg/(2*X.shape[0])

def gradienteLineal(X, Y, O, reg):    
    """la operacion que hace el gradiente por dentro -> devuelve un vector de valores"""  
    AuxO = np.hstack([np.zeros([1]), O[1:,]])
    O = O[np.newaxis]
    AuxO = AuxO[np.newaxis].T
    
    return ((X.T.dot(np.dot(X, O.T)-Y))/X.shape[0] + (reg/X.shape[0])*AuxO)

def minimizeFunc(O, X, Y, reg):
    return (costeLineal(X, Y, O, reg), gradienteLineal(X, Y, O, reg))

def main():
    # REGRESION LOGISTICA MULTICAPA
    valores = load_mat("ex5data1.mat")

    X = valores['X']         # datos de entrenamiento
    Y = valores['y']
    Xval = valores['Xval']   # ejemplos de validacion
    Yval = valores['yval']
    Xtest = valores['Xtest'] # prueba
    Ytest = valores['ytest']
    
    X = np.hstack([np.ones([X.shape[0], 1]), X])
    Xval = np.hstack([np.ones([Xval.shape[0], 1]), Xval])
    Xtest = np.hstack([np.ones([Xtest.shape[0], 1]), Xtest])

    m = X.shape[0]      # numero de muestras de entrenamiento
    n = X.shape[1]      # numero de variables x que influyen en el resultado y, mas la columna de 1s
    l = 0

    thetaVec = np.zeros([n])

    errorX = np.zeros(m - 1)
    errorXVal = np.zeros(m - 1)

    for i in range(1, m):
        result = opt.minimize(fun = minimizeFunc, x0 = thetaVec,
         args = (X[0:i], Y[0:i], l), method = 'TNC', jac = True, options = {'maxiter':70})
        O = result.x

        errorX[i-1] = costeLineal(X[0:i], Y[0:i], O, l)
        errorXVal[i-1] = costeLineal(Xval, Yval, O, l)

    functionGraphic(X, errorX, errorXVal)

main()