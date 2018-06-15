import pandas as pd
import numpy as np

### Link for Dataset - https://zenodo.org/record/841984#.WyNmtp9fi00 ###

df = pd.read_fwf('x_train.txt', header=None)
X_train = df[[0]]
df = pd.read_fwf('x_test.txt', header=None)
X_test = df[[0]]

target = pd.read_fwf('y_train.txt',header = None)
y_train = target[0]
target = pd.read_fwf('y_test.txt',header = None)
y_test = target[0]

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(y_train)
y_train = le.transform(y_train)
y_test = le.transform(y_test)

## function for main language coverage
def extra_lang_rem(X):
    
    for i in range(0, X.shape[0]):
        characters = np.empty(shape=(0,2))
        #counting characters for different paragraphs
        for char in X[0][i]:
            if len(characters) == 0:
                characters = np.append(characters,np.array([[char, 1]]), axis = 0)
            else:
                index, j =np.where(characters == char)
                if index.size and j.size:
                    characters[index[0]][1] = int(characters[index[0]][1]) + 1
                else:
                    characters = np.append(characters, np.array([[char, 1]]), axis = 0)
        # sort characters array descending by nc 
        characters = characters[(-(characters[:, 1].astype(np.int))).argsort()]
        
        # min coverage to delete unwanted characters
        min_coverage = 0.99
        C_theeta = np.array([])
        n_counter = 0
        total_char = np.sum(characters[:, 1].astype(np.int))
        for char in characters:
            n_counter += char[1].astype(np.int)
            if n_counter/ total_char < min_coverage:
                C_theeta = np.append(C_theeta, char[0])
        
        # remove unwanted characters
        for char in characters:
            index = np.where(char[0] == C_theeta)
            if not index[0].size:
                X[0][i] = X[0][i].replace(char[0], '')

    return X

## Evaluation Fucntion
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def print_score(clf, X_train, X_test, y_train, y_test, train = True):
    #     print accuracy score, classification report, confusion metrics
    if train:
#         training performance
        print('Train Result : \n')
        print('Accuracy Score {0:.4f}\n'.format(accuracy_score(y_train, clf.predict(X_train))))
        print('Classification Report : \n {} \n'.format(classification_report(y_train, clf.predict(X_train))))
        print('Confusion Metrics : \n {} \n'.format(confusion_matrix(y_train, clf.predict(X_train))))
        
        res = cross_val_score(clf, X_train, y_train, cv = 10, scoring='accuracy')
        print('Average Accuracy : {0:.4f}\n'.format(np.mean(res)))
        print('Accuracy SD : {0:.4f}\n'.format(np.std(res)))
        
    elif train == False:
#         test performance
        print('Test Result : \n')
        print('Accuracy Score {0:.4f}\n'.format(accuracy_score(y_test, clf.predict(X_test))))
        print('Classification Report : \n {}\n'.format(classification_report(y_test, clf.predict(X_test))))
        print('Confusion Metrics : \n {} \n'.format(confusion_matrix(y_test, clf.predict(X_test))))



X_train = extra_lang_rem(X_train)
X_test = extra_lang_rem(X_test)

## Word to vector Conversion using keras Tokenizer
from keras.preprocessing.text import Tokenizer
max_features = 1000000
tokenizer = Tokenizer(num_words=max_features, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', split=' ', lower=True)
tokenizer.fit_on_texts(X_train[0].values)
X_train = tokenizer.texts_to_sequences(X_train[0].values)
X_test = tokenizer.texts_to_sequences(X_test[0].values)

from keras.preprocessing.sequence import pad_sequences
X_train = pad_sequences(X_train, maxlen=500)
X_test = pad_sequences(X_test, maxlen=500)

## model designing
# using Naive Bayes
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X_train, y_train)
pred = clf.predict(X_test)

print_score(clf, X_train, X_test, y_train, y_test, train = True)
print_score(clf, X_train, X_test, y_train, y_test, train = False)
