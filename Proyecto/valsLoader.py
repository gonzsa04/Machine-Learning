import numpy as np
from dataReader import load_csv
from sklearn import preprocessing        # para polinomizar las Xs

def loadValues(file_name):
    valores = load_csv(file_name)

    totalX = valores[:, :valores.shape[1] - 1]   # datos de entrenamiento
    totalY = valores[:, valores.shape[1] - 1][np.newaxis].T
    #print(totalX.shape)
    #print(totalY.shape)

    Xlength = int(totalX.shape[0] * 0.6)
    XvalLength = int(totalX.shape[0] * 0.2)

    X = totalX[:Xlength, :]
    Y = totalY[:Xlength, :]
    #print(X.shape)
    #print(Y.shape)

    Xval = totalX[Xlength:XvalLength + Xlength, :]
    Yval = totalY[Xlength:XvalLength + Xlength, :]
    #print(Xval.shape)
    #print(Yval.shape)

    Xtest = totalX[XvalLength + Xlength:, :]
    Ytest = totalY[XvalLength + Xlength:, :]
    #print(Xtest.shape)
    #print(Ytest.shape)

    return X, Y, Xval, Yval, Xtest, Ytest

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