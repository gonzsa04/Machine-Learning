import numpy as np
import scipy.optimize as opt             # para la funcion de gradiente
from matplotlib import pyplot as plt     # para dibujar las graficas
from valsLoader import *
from dataReader import save_csv

# g(X*Ot) = h(x)
def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def dSigmoid(Z):
    return sigmoid(Z)*(1-sigmoid(Z))

def pesosAleatorios(L_in, L_out, rango):
    O = np.random.uniform(-rango, rango, (L_out, 1+L_in))
    return O

def cost(X, Y, O1, O2, reg):
    """devuelve un valor de coste"""

    a = -Y*(np.log(X))
    b = (1-Y)*(np.log(1-X))
    c = a - b
    d = (reg/(2*X.shape[0]))* ((O1[:,1:]**2).sum() + (O2[:,1:]**2).sum())
    return ((c.sum())/X.shape[0]) + d

def neuronalSuccessPercentage(results, Y):
    """determina el porcentaje de aciertos de la red neuronal comparando los resultados estimados con los resultados reales"""
    numAciertos = 0
    
    for i in range(results.shape[0]):
        result = np.argmax(results[i])
        if result == Y[i]: numAciertos += 1
        
    return (numAciertos/(results.shape[0]))*100

def forPropagation(X1, O1, O2):
    """propaga la red neuronal a traves de sus dos capas"""
    m = X1.shape[0]
    a1 = np.hstack([np.ones([m, 1]), X1])
    z2 = np.dot(a1, O1.T)
    a2 = np.hstack([np.ones([m, 1]), sigmoid(z2)])
    z3 = np.dot(a2, O2.T)
    h = sigmoid(z3)

    return a1, z2, a2, z3, h

def backPropAlgorithm(X, Y, O1, O2, num_etiquetas, reg):
    G1 = np.zeros(O1.shape)
    G2 = np.zeros(O2.shape)

    m = X.shape[0]
    a1, z2, a2, z3, h = forPropagation(X, O1, O2)

    for t in range(X.shape[0]):
        a1t = a1[t, :] # (1, 401)
        a2t = a2[t, :] # (1, 26)
        ht = h[t, :] # (1, 10)
        yt = Y[t] # (1, 10)
        d3t = ht - yt # (1, 10)
        d2t = np.dot(O2.T, d3t) * (a2t * (1 - a2t)) # (1, 26)

        G1 = G1 + np.dot(d2t[1:, np.newaxis], a1t[np.newaxis, :])
        G2 = G2 + np.dot(d3t[:, np.newaxis], a2t[np.newaxis, :])

    AuxO2 = O2
    AuxO2[:, 0] = 0

    G1 = G1/m
    G2 = G2/m + (reg/m)*AuxO2

    return np.concatenate((np.ravel(G1), np.ravel(G2)))


def backPropagation(params_rn, num_entradas, num_ocultas, num_etiquetas, X, Y, reg):    
    O1 = np.reshape(params_rn[:num_ocultas*(num_entradas + 1)], (num_ocultas, (num_entradas+1)))
    O2 = np.reshape(params_rn[num_ocultas*(num_entradas+1):], (num_etiquetas, (num_ocultas+1)))

    c = cost(forPropagation(X, O1, O2)[4], Y, O1, O2, reg)
    gradient = backPropAlgorithm(X,Y, O1, O2, num_etiquetas, reg)

    return c, gradient

def neuronalNetwork(X, Y, Xtest, Ytest, polyGrade):
    """aplica redes neuronales sobre un conjunto de datos, entrenando con una seccion de entrenamiento,
    y probando los resultados obtenidos (porcentaje de acierto) con una seccion de test"""

    poly = preprocessing.PolynomialFeatures(polyGrade)

    Xpoly = polynomize(X, polyGrade)                   # pone automaticamente columna de 1s
    Xnorm, mu, sigma = normalize(Xpoly[:, 1:]) # se pasa sin la columna de 1s (evitar division entre 0)

    XpolyTest = polynomize(Xtest, polyGrade)
    XnormTest = normalizeValues(XpolyTest[:, 1:], mu, sigma)

    m = Xnorm.shape[0]      # numero de muestras de entrenamiento
    n = Xnorm.shape[1]      # numero de variables x que influyen en el resultado y, mas la columna de 1s

    num_etiquetas = 2
    l = 1

    AuxY = np.zeros((m, num_etiquetas))

    for i in range(m):
        AuxY[i][int(Y[i])] = 1

    capaInter = 40
    O1 = pesosAleatorios(Xnorm.shape[1], capaInter, 0.12)
    O2 = pesosAleatorios(capaInter, num_etiquetas, 0.12)

    thetaVec = np.append(O1, O2).reshape(-1)

    result = opt.minimize(fun = backPropagation, x0 = thetaVec,
     args = (n, capaInter, num_etiquetas, Xnorm, AuxY, l), method = 'TNC', jac = True, options = {'maxiter':70})
    
    O1 = np.reshape(result.x[:capaInter*(n + 1)], (capaInter, (n+1)))
    O2 = np.reshape(result.x[capaInter*(n+1):], (num_etiquetas, (capaInter+1)))

    success = neuronalSuccessPercentage(forPropagation(XnormTest, O1, O2)[4], Ytest)
    print("Neuronal network success: " + str(success) + " %")

    return O1, O2, poly, success

def bestColumns(X, Y, Xtest, Ytest):
    """devuelve la mejor combinacion de columnas de X (las que obtienen un mejor porcentaje de acierto), mostrando
    el porcentaje de acierto obtenido de cada combinacion"""
    mostSuccessfull = 0
    bestRow = 0
    bestColumn = 0

    for x in range(0, 6):
        for y in range(0, 6):
            print("Row: " + str(x))
            print("Column: " + str(y))
            Xaux = np.vstack([X[:, x][np.newaxis], X[:, y][np.newaxis]]).T
            XtestAux = np.vstack([Xtest[:, x][np.newaxis], Xtest[:, y][np.newaxis]]).T

            O1, O2, poly, success = neuronalNetwork(Xaux, Y, XtestAux, Ytest, 3)
            if(success > mostSuccessfull):
                mostSuccessfull = success
                bestRow = x
                bestColumn = y

    return bestRow, bestColumn, mostSuccessfull

def main():
    X, Y, Xval, Yval, Xtest, Ytest = loadValues("steamReduced.csv")

    bestRow, bestColumn, mostSuccessfull = bestColumns(X, Y, Xtest, Ytest)
    # mejor combinacion de columnas (mejor porcentaje de acierto)
    print("Best Row: " + str(bestRow))
    print("Best Column: " + str(bestColumn))
    print("Most Successfull: " + str(mostSuccessfull))

    # redes neuronales con todas las columnas
    O1, O2, poly, success = neuronalNetwork(X, Y, Xtest, Ytest, 3)

    save_csv("O1.csv", O1)
    save_csv("O2.csv", O2)

#main()