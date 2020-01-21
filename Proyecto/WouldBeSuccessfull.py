from valsLoader import *
from dataReader import *
from NeuronalNetworks import forPropagation

def main():
    O1 = load_csv("O1.csv")
    O2 = load_csv("O2.csv")
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
    myX = myX.astype(np.float)

    print("Probabilidad de éxito: " + str(forPropagation(myX, O1, O2)[4][0,1] * 100) + "%")

main()