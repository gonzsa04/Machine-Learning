import numpy as np
from matplotlib import pyplot as plt     # para dibujar las graficas
from sklearn.svm import SVC
from valsLoader import *
from joblib import dump

def SVM(X, Y, Xval, Yval, Xtest, Ytest, polyGrade):   
    """aplica SVM sobre un conjunto de datos, entrenando con una seccion de entrenamiento,
    eligiendo las mejores C y sigma con una seccion de validacion, y probando los resultados obtenidos (porcentaje
    de acierto) con una seccion de test"""

    Xpoly = polynomize(X, polyGrade)                   # pone automaticamente columna de 1s
    Xnorm, mu, sigma = normalize(Xpoly[:, 1:]) # se pasa sin la columna de 1s (evitar division entre 0)

    XpolyVal = polynomize(Xval, polyGrade)
    XnormVal = normalizeValues(XpolyVal[:, 1:], mu, sigma)

    XpolyTest = polynomize(Xtest, polyGrade)
    XnormTest = normalizeValues(XpolyTest[:, 1:], mu, sigma)

    Y = Y.ravel()

    C = 0.01
    sigma = 0.01

    # mejores valores (resultado de probar lo de debajo, de esta forma no tenemos que ejecutarlo siempre)
    bestC = 66.0
    bestSigma = 2.5

    # probamos la mejor combinacion de C y sigma que de el menor error sobre los ejemplos de validacion
    """maxCorrects = 0
    for i in range(8):
        C = C * 3
        sigma = 0.01
        for j in range(8):
            sigma = sigma * 3
            clf = SVC(kernel='rbf', C=C, gamma= 1/(2*sigma**2))
            clf.fit(Xnorm, Y)
            corrects = (Yval[:, 0] == clf.predict(XnormVal)).sum()

            if maxCorrects < corrects:
                maxCorrects = corrects
                bestC = C
                bestSigma = sigma

    print(bestC)
    print(bestSigma)"""

    clf = SVC(kernel='rbf', C=bestC, gamma= 1/(2*bestSigma**2))
    clf.fit(Xnorm, Y)
    
    corrects = (Ytest[:, 0] == clf.predict(XnormTest)).sum()
    print((corrects / Xtest.shape[0])*100)

    return clf

def main():
    X, Y, Xval, Yval, Xtest, Ytest = loadValues("steamReduced.csv")

    clf = SVM(X, Y, Xval, Yval, Xtest, Ytest, 1)
    dump(clf, 'clf.joblib') 

main()