import numpy as np
import scipy.optimize as opt             # para la funcion de gradiente
import displayData
from matplotlib import pyplot as plt     # para dibujar las graficas
from scipy.io import loadmat

def load_mat(file_name):
    """carga el fichero mat especificado y lo devuelve en una matriz data"""
    return loadmat(file_name)

def graphics(X):
    """Selecciona aleatoriamente 10 ejemplos y los pinta"""
    sample = np.random.choice(X.shape[0], 100)
    displayData.displayData(X[sample, :])
    plt.show()

# g(X*Ot) = h(x)
def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def cost(X, Y, reg):
    """devuelve un valor de coste"""
    a = -Y*(np.log(X))
    b = (1-Y)*(np.log(1-X))
    c = a - b
    return c.sum()/X.shape[0]
    #return (-np.log(X).T.dot(Y) - np.log(1-X).T.dot(1-Y)).sum()/X.shape[0]
    #+ (reg/(2*X.shape[0]))*(O[1:,]**2).sum()) # igual que el coste anterior pero añadiendole esto ultimo para la regularizacion

def gradient(O, X, Y, l):    
    """la operacion que hace el gradiente por dentro -> devuelve un vector de valores"""  
    AuxO = np.hstack([np.zeros([1]), O[1:,]]) # sustituimos el primer valor de las thetas por 0 para que el termino independiente
    # no se vea afectado por la regularizacion (lambda*0 = 0)
    return (((X.T.dot(sigmoid(X.dot(O))-np.ravel(Y)))/X.shape[0])
    + (l/X.shape[0])*AuxO) # igual que el gradiente anterior pero añadiendole esto ultimo para la regularizacion

def neuronalSuccessPercentage(results, Y):
    """determina el porcentaje de aciertos de la red neuronal comparando los resultados estimados con los resultados reales"""
    numAciertos = 0
    
    for i in range(results.shape[0]):
        result = np.argmax(results[i]) + 1
        if result == Y[i]: numAciertos += 1
    return (numAciertos/(results.shape[0]))*100

def forPropagation(X1, O1, O2):
    """propaga la red neuronal a traves de sus dos capas"""
    X2 = sigmoid(X1.dot(O1.T))
    X2 = np.hstack([np.ones([X2.shape[0], 1]), X2])
    return sigmoid(X2.dot(O2.T))

def backPropagation(params_rn, num_entradas, num_ocultas, num_etiquetas, X, Y, reg):    
    AuxY = np.zeros([Y.shape[0], num_etiquetas])
    Y[Y == 10] = 0 
    for i in range(Y.shape[0]):
        if Y[i] == 0: AuxY[9] = 1
        else: AuxY[i, Y[i]-1] = 1

    O1 = np.reshape(params_rn[:num_ocultas*(num_entradas + 1)], (num_ocultas, (num_entradas+1)))
    O2 = np.reshape(params_rn[num_ocultas*(num_entradas+1):], (num_etiquetas, (num_ocultas+1)))

    result = forPropagation(X, O1, O2)

    print(cost(result, AuxY, reg))


def main():
    # REGRESION LOGISTICA MULTICAPA
    valores = load_mat("ex4data1.mat")

    X = valores['X'] # matriz X, con todas las filas y todas las columnas menos la ultima (ys)
    Y = valores['y'] # matriz Y, con todas las filas y la ultima columna
    
    m = X.shape[0]      # numero de muestras de entrenamiento
    n = X.shape[1]      # numero de variables x que influyen en el resultado y, mas la columna de 1s
    num_etiquetas = 10
    l = 0.1

    X = np.hstack([np.ones([X.shape[0], 1]), X])

    # REDES NEURONALES
    weights = load_mat('ex4weights.mat')
    O1, O2 = weights['Theta1'], weights['Theta2']

    thetaVec = np.concatenate((np.ravel(O1), np.ravel(O2)))[np.newaxis]

    backPropagation(thetaVec.T, n, 25, num_etiquetas, X, Y, l)

    #success = neuronalSuccessPercentage(forPropagation(X, O1, O2), Y)
    #print("Neuronal network success: " + str(success) + " %")

    #GRAFICAS
    graphics(X[:, 1:])

main()