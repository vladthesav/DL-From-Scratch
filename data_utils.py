import zipfile
import pandas as pd
import numpy as np
import os

if 'mnist_train_small.csv' not in os.listdir('data'):
    print('mnist_train_small.csv not detected: will extract zip')

    with zipfile.ZipFile('data/mnist_train_small.zip', 'r') as zip_ref:
        zip_ref.extractall('data/mnist_train_small.csv')

def one_hot_list(Y):
    return list(set(Y))

def one_hot(s, labels):
  out = np.zeros(len(labels))
  out[s] = 1
  return out

class MNISTData():
    """this is an interface to load MNIST data"""

    def __init__(self, path = 'data/mnist_train_small.csv', flatten = False):
        self.path = path

        raw_df = pd.read_csv(path)

        self.labels = one_hot_list(raw_df.iloc[:, 0])

        X_raw= raw_df.iloc[:, 1:].values

        mean, std = np.mean(X_raw), np.std(X_raw)

        normalize = lambda x : (x-mean)/std
        reshape = lambda z: z.reshape((28,28))

        X = []
        for x in X_raw:
            if not flatten:
                X.append(normalize(x))
            else:
                x.append(reshape(normalize(x)))

        self.x = np.array(X)


        Y_raw = raw_df.iloc[:, 0]
        Y = []
        for y in Y_raw.values:
            Y.append(one_hot(y, self.labels))
        
        self.y = np.array(Y)



        


#mnist = MNISTData()
#print(mnist.labels)



