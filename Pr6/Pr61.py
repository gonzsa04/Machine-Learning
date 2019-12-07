import numpy as np
from matplotlib import pyplot as plt     # para dibujar las graficas
from scipy.io import loadmat
from sklearn.svm import SVC

def load_mat(file_name):
    """carga el fichero mat especificado y lo devuelve en una matriz data"""
    return loadmat(file_name)

def functionGraphic(X, Y, clf):
    # pintamos muestras de entrenamiento

    pos = np.where(Y==1)[0]
    plt.scatter(X[pos, 0], X[pos, 1], marker='+', c= 'k')
    
    pos = np.where(Y==0)[0]
    plt.scatter(X[pos, 0], X[pos, 1], marker='o', c= 'y')

    
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)

    # plot decision boundary and margins
    ax.contour(XX, YY, Z, colors='k', levels=[0]
    #, alpha=0.5, linestyles=['--', '-', '--']
    )

    # plot support vectors
    #ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
    #        linewidth=1, facecolors='none', edgecolors='k')

    plt.savefig('TrainingExamples.png')# plot the decision function

    plt.show()

def main():
    valores = load_mat("ex6data3.mat")

    X = valores['X']         # datos de entrenamiento
    y = valores['y']

    Xval = valores['Xval']         # datos de validacion
    yval = valores['yval']

    C = 0.01
    sigma = 0.01

    maxCorrects = 0
    bestC = C
    bestSigma = sigma

    # probamos la mejor combinacion de C y sigma que de el menor error
    for i in range(8):
        C = C * 3
        sigma = 0.1
        for j in range(8):
            sigma = sigma * 3
            clf = SVC(kernel='rbf', C=C, gamma= 1/(2*sigma**2))
            clf.fit(X, y)
            corrects = (yval[:, 0] == clf.predict(Xval)).sum()

            if maxCorrects < corrects:
                maxCorrects = corrects
                bestC = C
                bestSigma = sigma
    
    clf = SVC(kernel='rbf', C=bestC, gamma= 1/(2*bestSigma**2))
    clf.fit(X, y)

    functionGraphic(X, y, clf)

main()