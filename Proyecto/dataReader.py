from pandas.io.parsers import read_csv   # para leer .csv
import numpy as np

def load_csv(file_name):
    """carga el fichero csv especificado y lo devuelve en un array de numpy"""
    valores = read_csv(file_name, header=None).values

    return valores

def reduceData(data, columns):
    for i in range(columns.shape[0]):
        data = np.delete(data, columns[i], 1)

    return data

def dataToNumbers(data):
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):

            # release date
            if j == 0:
                data[i, j] = data[i, j][:4]

            # platform
            elif j == 2:
                if data[i, j] == 'windows;mac;linux':
                    data[i, j] = 1
                elif data[i, j] == 'windows':
                    data[i, j] = 2
                else:
                    data[i, j] = 3

            # category
            elif j == 4:
                index = data[i,j].find('Single-Player')
                if index != -1:
                    data[i, j] = 1
                else:
                    data[i, j] = 2

            # genre
            elif j == 5:
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
                
                data[i, j] = aux
                #Action Strategy RPG Casual Simulation Racing Adventure Sports
                
            # owners mean
            elif j == 10:     
                index = data[i,j].find('-')
                minimum = float(data[i,j][:index])
                maximum = float(data[i,j][index+1:])
                data[i, j] = (minimum + maximum)/2

            data[i, j] = float(data[i, j])
    
    return data


def main():
    X = load_csv("steam.csv")
    X = reduceData(X, np.array([11, 10, 5, 4, 1, 0]))
    tags = X[0, :]
    X = np.delete(X, 0, 0)

    X = dataToNumbers(X)
    print(tags)
    print(X[24,:])

main()

#0 1 4 5 10 11 