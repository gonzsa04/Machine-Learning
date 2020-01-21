import numpy as np
import scipy.optimize as opt             # para la funcion de gradiente
from matplotlib import pyplot as plt     # para dibujar las graficas
from valsLoader import *

def graphics(X, Y, O, poly):
    plot_decisionboundary(X, Y, O, poly)

    # Obtiene un vector con los índices de los ejemplos positivos
    pos = np.where(Y == 1)
    # Dibuja los ejemplos positivos
    plt.scatter(X[pos, 0] ,X[pos, 1] , c = 'blue', s=2)

    # Obtiene un vector con los índices de los ejemplos negativos
    pos = np.where(Y == 0)
    # Dibuja los ejemplos negativos
    plt.scatter(X[pos, 0] ,X[pos, 1] , c = 'red', s=2)

    plt.show()

def plot_decisionboundary(X, Y, theta, poly):
    """pinta el polinomio que separa los datos entre los que cumplen el requisito y los que no"""
    plt.figure()

    x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max()

    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max),
    np.linspace(x2_min, x2_max)) # grid de cada columna de Xs

    h = sigmoid(poly.fit_transform(np.c_[xx1.ravel(), xx2.ravel()]).dot(theta))# ravel las pone una tras otra

    h = h.reshape(xx1.shape)

    plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='g')

# g(X*Ot) = h(x)
def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def lambdaGraphic(errorX, errorXVal, lambdas):
    """pinta la curva de aprendizaje de lambda"""
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

def bestColumns(X, Y, Xval, Yval, Xtest, Ytest):
    """devuelve la mejor combinacion de columnas de X (las que obtienen un mejor porcentaje de acierto), mostrando
    la grafica y el porcentaje de acierto obtenido de cada combinacion"""
    mostSuccessfull = 0
    bestRow = 0
    bestColumn = 0

    for x in range(0, 6):
        for y in range(0, 6):
            print("Row: " + str(x))
            print("Column: " + str(y))
            Xaux = np.vstack([X[:, x][np.newaxis], X[:, y][np.newaxis]]).T
            XvalAux = np.vstack([Xval[:, x][np.newaxis], Xval[:, y][np.newaxis]]).T
            XtestAux = np.vstack([Xtest[:, x][np.newaxis], Xtest[:, y][np.newaxis]]).T
            samples = np.random.choice(X.shape[0], 400)

            O, poly, success = logisticRegresion(Xaux, Y, XvalAux, Yval, XtestAux, Ytest, 6)
            if(success > mostSuccessfull):
                mostSuccessfull = success
                bestRow = x
                bestColumn = y

            Xaux = Xaux[samples,:]
            Yaux = Y[samples]
            print(Xaux.shape)
            graphics(Xaux, Yaux, O, poly)

    return bestRow, bestColumn, mostSuccessfull

def logisticRegresion(X, Y, Xval, Yval, Xtest, Ytest, polyGrade):
    """aplica regresion logistica sobre un conjunto de datos, entrenando con una seccion de entrenamiento,
    eligiendo la mejor lambda con una seccion de validacion, y probando los resultados obtenidos (porcentaje
    de acierto) con una seccion de test"""
    
    poly = preprocessing.PolynomialFeatures(polyGrade)

    Xpoly = polynomize(X, polyGrade)                   # pone automaticamente columna de 1s
    Xnorm, mu, sigma = normalize(Xpoly[:, 1:]) # se pasa sin la columna de 1s (evitar division entre 0)
    Xnorm = np.hstack([np.ones([Xnorm.shape[0], 1]), Xnorm]) # volvemos a poner columna de 1s

    XpolyVal = polynomize(Xval, polyGrade)
    XnormVal = normalizeValues(XpolyVal[:, 1:], mu, sigma)
    XnormVal = np.hstack([np.ones([XnormVal.shape[0], 1]), XnormVal])

    XpolyTest = polynomize(Xtest, polyGrade)
    XnormTest = normalizeValues(XpolyTest[:, 1:], mu, sigma)
    XnormTest = np.hstack([np.ones([XnormTest.shape[0], 1]), XnormTest])

    m = Xnorm.shape[0]      # numero de muestras de entrenamiento
    n = Xnorm.shape[1]      # numero de variables x que influyen en el resultado y, mas la columna de 1s

    thetaVec = np.zeros([n])

    l =  np.arange(0, 3, 0.1)

    errorX = np.zeros(l.shape[0])
    errorXVal = np.zeros(l.shape[0])

    # errores para cada valor de lambda
    for i in range(l.shape[0]):
        result = opt.minimize(fun = minimizeFunc, x0 = thetaVec,
        args = (Xnorm, Y, l[i]), method = 'TNC', jac = True, options = {'maxiter':70})
        O = result.x

        errorX[i] = coste(Xnorm, Y, O, l[i])
        errorXVal[i] = coste(XnormVal, Yval, O, l[i])

    lambdaGraphic(errorX, errorXVal, l)

    # lambda que hace el error minimo en los ejemplos de validacion
    lambdaIndex = np.argmin(errorXVal)
    print("Best lambda: " + str(l[lambdaIndex]))

    # thetas usando la lambda que hace el error minimo (sobre ejemplos de entrenamiento)
    result = opt.minimize(fun = minimizeFunc, x0 = thetaVec,
        args = (Xnorm, Y, l[lambdaIndex]), method = 'TNC', jac = True, options = {'maxiter':70})

    O = result.x
            
    success = successPercentage(XnormTest, Ytest, O)
    print("Porcentaje de acierto: " + str(success*100) + "%")

    return O, poly, success

def main():
    # dataset
    X, Y, Xval, Yval, Xtest, Ytest = loadValues("steamReduced.csv")

    bestRow, bestColumn, mostSuccessfull = bestColumns(X, Y, Xval, Yval, Xtest, Ytest)
    # mejor combinacion de columnas (mejor porcentaje de acierto)
    print("Best Row: " + str(bestRow))
    print("Best Column: " + str(bestColumn))
    print("Most Successfull: " + str(mostSuccessfull*100))

    # regresion logistica con todas las columnas
    logisticRegresion(X, Y, Xval, Yval, Xtest, Ytest, 2)

main()