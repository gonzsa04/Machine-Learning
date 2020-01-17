import numpy as np
from matplotlib import pyplot as plt     # para dibujar las graficas
from sklearn.svm import SVC
from valsLoader import *
from dataReader import load_mat

def functionGraphic(X, Y, clf):
    # pintamos muestras de entrenamiento

    pos = np.where(Y==1)[0]
    plt.scatter(X[pos, 0], X[pos, 1], marker='+', c= 'k', s=5)
    
    pos = np.where(Y==0)[0]
    plt.scatter(X[pos, 0], X[pos, 1], marker='o', c= 'y', s=5)

    ax = plt.gca()

    # create grid to evaluate model
    xx = np.arange(0, 160, 0.5)
    yy = np.arange(0, 50, 0.5)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)

    # plot decision boundary and margins
    ax.contour(XX, YY, Z, colors='k', levels=[0]
    , alpha=0.5, linestyles=['--', '-', '--']
    )

    # plot support vectors
    #ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
    #        linewidth=1, facecolors='none', edgecolors='k')

    plt.savefig('TrainingExamples.png')# plot the decision function

    plt.show()

def main():
    X, Y, Xval, Yval, Xtest, Ytest = loadValues("steamReduced.csv")

    polyGrade = 1

    Xpoly = polynomize(X, polyGrade)                   # pone automaticamente columna de 1s
    Xnorm, mu, sigma = normalize(Xpoly[:, 1:]) # se pasa sin la columna de 1s (evitar division entre 0)

    XpolyVal = polynomize(Xval, polyGrade)
    XnormVal = normalizeValues(XpolyVal[:, 1:], mu, sigma)

    XpolyTest = polynomize(Xtest, polyGrade)
    XnormTest = normalizeValues(XpolyTest[:, 1:], mu, sigma)

    Y = Y.ravel()

    C = 0.01
    sigma = 0.01

    bestC = 66.0
    bestSigma = 2.5

    """# probamos la mejor combinacion de C y sigma que de el menor error
    maxCorrects = 0
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

    #debug
    x = 2
    y = 5
    X = np.vstack([X[:, x][np.newaxis], X[:, y][np.newaxis]]).T
    Xnorm = np.vstack([Xnorm[:, x][np.newaxis], Xnorm[:, y][np.newaxis]]).T
    XnormTest = np.vstack([XnormTest[:, x][np.newaxis], XnormTest[:, y][np.newaxis]]).T
    samples = np.random.choice(X.shape[0], 400)

    clf.fit(Xnorm, Y)

    corrects = (Ytest[:, 0] == clf.predict(XnormTest)).sum()

    print((corrects / Xtest.shape[0])*100)

    X = X[samples,:]
    Y = Y[samples]
    functionGraphic(X, Y, clf)

#0 5, 2 0, 2 5, 3 4

main()