import numpy as np
import scipy.optimize as opt             # para la funcion de gradiente
from matplotlib import pyplot as plt     # para dibujar las graficas
from scipy.io import loadmat
from sklearn import preprocessing        # para polinomizar las Xs

def load_mat(file_name):
    """carga el fichero mat especificado y lo devuelve en una matriz data"""
    return loadmat(file_name)

def hLine(X, O):
    """devuelve la funcion h(x) usando la ecuacion de la recta"""
    return O[0] + O[1]*X

def functionGraphic(X, Y, O, mu, sigma):
    minValue = np.amin(X)
    maxValue = np.amax(X)
    x = np.linspace(minValue, maxValue, 256, endpoint=True)[np.newaxis].T
    xPoly = polynomize(x, 8)
    xNorm = normalizeValues(xPoly[:, 1:], mu, sigma)
    xNorm = np.hstack([np.ones([xNorm.shape[0], 1]), xNorm])

    y = np.dot(xNorm, O)

    # pintamos muestras de entrenamiento
    plt.scatter(X[:, 0], Y, 1, 'red')
    # pintamos funcion de estimacion
    plt.plot(x, y)
    plt.savefig('PolynomialH(X).png')
    plt.show()

def learningCurve(errorX, errorXVal):
    """muestra el grafico de la funcion h(x)"""

    x = np.linspace(0, 12, errorX.shape[0], endpoint=True)
    xVal = np.linspace(0, 12, errorX.shape[0], endpoint=True)

    # pintamos funcion de estimacion
    plt.plot(x, errorX)
    plt.plot(xVal, errorXVal)
    plt.savefig('LearningCurvePolyLambda0.png')
    plt.show()

def costeLineal(X, Y, O, reg):
    """devuelve la funcion de coste, dadas X, Y, y thetas"""
    O = O[np.newaxis]
    AuxO = O[:, 1:]

    H = np.dot(X, O.T)
    Aux = (H-Y)**2
    cost = Aux.sum()/(2*len(X))

    return cost + (AuxO**2).sum()*reg/(2*X.shape[0])

def gradienteLineal(X, Y, O, reg):    
    """la operacion que hace el gradiente por dentro -> devuelve un vector de valores"""  
    AuxO = np.hstack([np.zeros([1]), O[1:,]])
    O = O[np.newaxis]
    AuxO = AuxO[np.newaxis].T
    
    return ((X.T.dot(np.dot(X, O.T)-Y))/X.shape[0] + (reg/X.shape[0])*AuxO)

def minimizeFunc(O, X, Y, reg):
    return (costeLineal(X, Y, O, reg), gradienteLineal(X, Y, O, reg))

def polynomize(X, p):
    poly = preprocessing.PolynomialFeatures(p)
    return poly.fit_transform(X) # añade automaticamente la columna de 1s

def normalize(X):
    """normalizacion de escalas, para cuando haya mas de un atributo"""
    mu = X.mean(0)[np.newaxis]   # media de cada columna de X
    sigma = X.std(0)[np.newaxis] # desviacion estandar de cada columna de X

    X_norm = (X - mu)/sigma

    return X_norm, mu, sigma

def normalizeValues(valoresPrueba, mu, sigma):
    """normaliza los valores de prueba con la mu y sigma de los atributos X (al normalizarlos)"""
    return (valoresPrueba - mu)/sigma

def main():
    # REGRESION LOGISTICA MULTICAPA
    valores = load_mat("ex5data1.mat")

    X = valores['X']         # datos de entrenamiento
    Y = valores['y']
    Xval = valores['Xval']   # ejemplos de validacion
    Yval = valores['yval']
    Xtest = valores['Xtest'] # prueba
    Ytest = valores['ytest']
    
    #X = np.hstack([np.ones([X.shape[0], 1]), X])
    #Xval = np.hstack([np.ones([Xval.shape[0], 1]), Xval])
    Xtest = np.hstack([np.ones([Xtest.shape[0], 1]), Xtest])

    Xpoly = polynomize(X, 8)
    Xnorm, mu, sigma = normalize(Xpoly[:, 1:])
    Xnorm = np.hstack([np.ones([Xnorm.shape[0], 1]), Xnorm])

    XpolyVal = polynomize(Xval, 8)
    XnormVal, muVal, sigmaVal = normalize(XpolyVal[:, 1:])
    XnormVal = np.hstack([np.ones([XnormVal.shape[0], 1]), XnormVal])

    m = Xnorm.shape[0]      # numero de muestras de entrenamiento
    n = Xnorm.shape[1]      # numero de variables x que influyen en el resultado y, mas la columna de 1s
    l = 0

    thetaVec = np.zeros([n])

    errorX = np.zeros(m - 1)
    errorXVal = np.zeros(m - 1)

    result = opt.minimize(fun = minimizeFunc, x0 = thetaVec,
         args = (Xnorm, Y, l), method = 'TNC', jac = True, options = {'maxiter':70})

    O = result.x

    functionGraphic(X, Y, O, mu, sigma)

    for i in range(1, m):
        result = opt.minimize(fun = minimizeFunc, x0 = thetaVec,
         args = (Xnorm[0:i], Y[0:i], l), method = 'TNC', jac = True, options = {'maxiter':70})
        O = result.x

        errorX[i-1] = costeLineal(Xnorm[0:i], Y[0:i], O, l)
        errorXVal[i-1] = costeLineal(XnormVal, Yval, O, l)

    learningCurve(errorX, errorXVal)

main()