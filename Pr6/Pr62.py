import numpy as np
from sklearn.svm import SVC
from process_email import email2TokenList 
import codecs
from get_vocab_dict import getVocabDict 

def emailToWordOcurrence(email, wordsDict):
    result = np.zeros(len(wordsDict))

    for i in range(len(email)):
        if email[i] in wordsDict:
            index = wordsDict.get(email[i]) - 1
            result[index] = 1
    return result

def dataManager(ini, fin, directoryName, yValue):
    X = np.empty((0, 1899)) # 60% de 500
    Y = np.empty((0, 1))

    for i in range(ini + 1, fin + 1): 
        email_contents = codecs.open('{0}/{1:04d}.txt'.format(directoryName, i), 'r',
        encoding='utf-8', errors='ignore').read()
        email = email2TokenList(email_contents)

        wordsDict = getVocabDict()

        wordOcurrence = emailToWordOcurrence(email, wordsDict)
        X = np.vstack((X, wordOcurrence))
        Y = np.vstack((Y, yValue))
    
    return X, Y

def main():
    Xspam, Yspam = dataManager(0, 300, "spam", 1)
    Xeasy, Yeasy = dataManager(0, 1531, "easy_ham", 0)
    Xhard, Yhard = dataManager(0, 150, "hard_ham", 0)

    X = np.vstack((Xspam, Xeasy))
    X = np.vstack((X, Xhard))
    Y = np.vstack((Yspam, Yeasy))
    Y = np.vstack((Y, Yhard))

    XspamVal, YspamVal = dataManager(300, 400, "spam", 1)
    XeasyVal, YeasyVal = dataManager(1531, 2041, "easy_ham", 0)
    XhardVal, YhardVal = dataManager(150, 200, "hard_ham", 0)
    
    Xval = np.vstack((XspamVal, XeasyVal))
    Xval = np.vstack((Xval, XhardVal))
    Yval = np.vstack((YspamVal, YeasyVal))
    Yval = np.vstack((Yval, YhardVal))

    XspamTest, YspamTest = dataManager(400, 500, "spam", 1)
    XeasyTest, YeasyTest = dataManager(2041, 2551, "easy_ham", 0)
    XhardTest, YhardTest = dataManager(200, 250, "hard_ham", 0)

    Xtest = np.vstack((XspamTest, XeasyTest))
    Xtest = np.vstack((Xtest, XhardTest))
    Ytest = np.vstack((YspamTest, YeasyTest))
    Ytest = np.vstack((Ytest, YhardTest))

    # mejores C y sigma aplicando el metodo de la parte anterior
    # solo lo hacemos una vez y nos guardamos los valores -> tarda mucho
    bestC = 21.87
    bestSigma = 8.1
    
    clf = SVC(kernel='rbf', C=bestC, gamma= 1/(2*bestSigma**2))
    clf.fit(X, Y)

    corrects = (Ytest[:, 0] == clf.predict(Xtest)).sum()

    print((corrects / Xtest.shape[0])*100)

main()