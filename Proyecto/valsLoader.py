import numpy as np
from dataReader import load_csv
from sklearn import preprocessing        # para polinomizar las Xs

def loadValues(file_name):
    """lee valores y los procesa, dividiendolos en datos de entrenamiento, validacion y test"""

    valores = load_csv(file_name)

    totalX = valores[:, :valores.shape[1] - 1]
    totalY = valores[:, valores.shape[1] - 1][np.newaxis].T

    Xlength = int(totalX.shape[0] * 0.6)
    XvalLength = int(totalX.shape[0] * 0.2)

    X = totalX[:Xlength, :]
    Y = totalY[:Xlength, :]

    Xval = totalX[Xlength:XvalLength + Xlength, :]
    Yval = totalY[Xlength:XvalLength + Xlength, :]

    Xtest = totalX[XvalLength + Xlength:, :]
    Ytest = totalY[XvalLength + Xlength:, :]

    return X, Y, Xval, Yval, Xtest, Ytest

def polynomize(X, p):
    """polinomiza X con grado p"""
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