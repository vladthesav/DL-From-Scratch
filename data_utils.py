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

def one_hot_(Y):
    return None


class MNISTData():
    """this is an interface to load MNIST data"""

    def __init__(self, path = 'data/mnist_train_small.csv'):
        self.path = path
        raw_df = pd.read_csv(path)
        self.labels = one_hot_list(raw_df.iloc[:, 0])
        self.X_raw= raw_df.iloc[:, 1:]
        self.Y_raw = raw_df.iloc[:, 0]
        


mnist = MNISTData()
print(mnist.labels)



