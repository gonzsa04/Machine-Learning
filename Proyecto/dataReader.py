from pandas.io.parsers import read_csv   # para leer .csv
import numpy as np
from scipy.io import loadmat

def load_csv(file_name):
    """carga el fichero csv especificado y lo devuelve en un array de numpy"""
    return read_csv(file_name, header=None).values

def save_csv(file_name, array):
    """guarda el array especificado y lo vuelca como csv"""
    np.savetxt(file_name, array, delimiter=",")

def reduceData(data, columns):
    """quita de "data" las columnas establecidas por "columns" """
    for i in range(columns.shape[0]):
        data = np.delete(data, columns[i], 1)

    return data

def dataToNumbers(data):
    """ pasa todos los datos que no sean numeros a valores numericos"""

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):

            # release date
            if j == 0:
                data[i, j] = data[i, j][:4]

            # platform
            elif j == 1:
                if data[i, j] == 'windows;mac;linux':
                    data[i, j] = 1
                elif data[i, j] == 'windows':
                    data[i, j] = 2
                else:
                    data[i, j] = 3

            # genre
            elif j == 2:
                aux = 0 
                if data[i,j].find('Action') != -1:
                    aux += 7
                if data[i,j].find('Strategy') != -1:
                    aux += 11
                if data[i,j].find('RPG') != -1:
                    aux += 13
                if data[i,j].find('Casual') != -1:
                    aux += 17
                if data[i,j].find('Simulation') != -1:
                    aux += 19
                if data[i,j].find('Racing') != -1:
                    aux += 23
                if data[i,j].find('Adventure') != -1:
                    aux += 29
                if data[i,j].find('Sports') != -1:
                    aux += 31
                if data[i,j].find('Indie') != -1:
                    aux += 37
                
                data[i, j] = aux
                #Action Strategy RPG Casual Simulation Racing Adventure Sports Indie
                
            # owners mean
            elif j == 5:     
                index = data[i,j].find('-')
                minimum = float(data[i,j][:index])
                maximum = float(data[i,j][index+1:])
                data[i, j] = (minimum + maximum)/2

            data[i, j] = float(data[i, j])
    
    return data

def createY(owners):
    """crea las Ys sobre los datos de los propietarios: si el juego tiene mas
    propietarios que la media de propietarios, se considera exitoso (y = 1); si no,
    se considera no exitoso (y = 0)"""

    L = owners.shape[0]
    mean = owners.sum()/L
    aux = (owners > mean)
    return aux.astype(int) 

def main():
    X = load_csv("steam.csv")
    #columnas que desechamos
    X = reduceData(X, np.array([15, 14, 11, 10, 8, 7, 5, 4, 3, 1, 0]))
    
    #print de todos los tags (lo que significa cada columna)
    tags = X[0, :]
    print(tags)
    
    X = np.delete(X, 0, 0) # quitamos tags
    X = dataToNumbers(X)
    X = np.random.permutation(X) # desordenamos aleatoriamente
    Y = createY(X[:, 5])[np.newaxis].T
    X = np.delete(X, 5, 1) # borramos columna de owners
    X = np.append(X, Y, 1) # añadimos la transformacion (Ys) al final

    save_csv("steamReduced.csv", X)

#main()

#release date, platforms, genre, positives, negatives, price
#owners
