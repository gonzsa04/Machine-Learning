import numpy as np
from sklearn.svm import SVC
from process_email import email2TokenList 
import codecs
from get_vocab_dict import getVocabDict 

def emailToWordOcurrence(email, wordsDict):
    result = np.zeros(len(wordsDict))

    for i in range(len(wordsDict)):
        if wordsDict[i][0] in email:
            result[i] = 1

    return result

def main():
    email_contents = codecs.open('spam/0001.txt', 'r', encoding='utf-8', errors='ignore').read()
    email = email2TokenList(email_contents)

    wordsDict = getVocabDict()

    wordOcurrence = emailToWordOcurrence(email, wordsDict)
    print(wordOcurrence)

main()