from valsLoader import *
from dataReader import dataToNumbers
from joblib import load

def main():
    X, Y, Xval, Yval, Xtest, Ytest = loadValues("steamReduced.csv")

    clf = load('clf.joblib') 
    myX = np.array([])

    print("Introduce los parámetros de tu juego: ")

    print("Fecha de salida (año): ")
    myX = np.append(myX, input())
    print("Plataforma (windows;mac;linux, windows): ")
    myX = np.append(myX, input())
    print("Género (Action, Strategy, RPG, Casual, Simulation, Racing, Adventure, Sports, Indie): ")
    myX = np.append(myX, input())
    print("Votos positivos: ")
    myX = np.append(myX, input())
    print("Votos negativos: ")
    myX = np.append(myX, input())
    myX = np.append(myX, '0.0')
    print("Precio (euros): ")
    myX = np.append(myX, input())

    myX = myX[np.newaxis]
    myX = dataToNumbers(myX)
    myX = np.delete(myX, 5, 1) # borramos columna de owners
    print(clf.predict(myX))

    #print((corrects / Xtest.shape[0])*100)

main()