# -*- coding: utf-8 -*-
'''
A naive version of the SVM-KNN is: for a query,

1. compute distances of the query to all training examples and pick the nearest K neighbors;

2. if the K neighbors have all the same labels, the query is labeled and exit; else, compute the pairwise distances between the K neighbors;

3. convert the distance matrix to a kernel matrix and apply multiclass SVM;

4. use the resulting classifier to label the query.
'''

import datetime
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn import neighbors
from sklearn.externals import joblib
from sklearn.metrics import classification_report, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC

from dataprocessing import drawdata, readfile
from KNN import knn_train


def file2np(file_name):
    xList = []
    yList = []
    read_file = open(file_name, 'r')
    for line in read_file.read().split('\n'):
        if line != '':
            yList.append(float(line.split(' ')[0]))
            xList.append([float(tk.split(':')[1])
                         for tk in line.split(' ')[1:]])
    xnp = np.array(xList)
    ynp = np.array(yList)
    return xnp, ynp


def nnsvm_train(x_train, y_train, x_test, y_test, numk):
    '''
        p : integer, optional (default = 2)
        Power parameter for the Minkowski metric. When p = 1, this is
        equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.
    '''

    knn = neighbors.NearestNeighbors(n_neighbors=numk, algorithm='auto', p=2)
    knn.fit(x_train)

    # initialization
    length_train = len(y_train)
    length_test = len(y_test)
    y_hat_train = np.zeros_like(y_train)
    y_hat_test = np.zeros_like(y_test)

    # knn_train
    nb = knn.kneighbors(x_train, return_distance=False)

    # Naive SVM_KNN_train
    for i in range(length_train):
        base = y_train[nb[i][0]]
        if all(y_train[j] == base for j in nb[i]):
            y_hat_train[i] = base
        else:
            x_temp = x_train[nb[i]]
            y_temp = y_train[nb[i]]
            clf = LinearSVC()
            clf.fit(x_temp, y_temp)
            y_hat_train[i] = clf.predict(x_train[i].reshape(1, -1))[0]

    # result_train
    train_error = float((y_hat_train != y_train).mean())
    

    # knn_test
    nb = knn.kneighbors(x_test, return_distance=False)

    # Naive SVM_KNN_test
    for i in range(length_test):
        base = y_train[nb[i][0]]
        if all(y_train[j] == base for j in nb[i]):
            y_hat_test[i] = base
        else:
            x_temp = x_train[nb[i]]
            y_temp = y_train[nb[i]]
            clf = LinearSVC()
            clf.fit(x_temp, y_temp)
            y_hat_test[i] = clf.predict(x_test[i].reshape(1, -1))[0]
    
    # result_test
    test_error = float((y_hat_test != y_test).mean())

    result = {
        'train_error': train_error,
        'test_error': test_error,
    }
    return result

def training(ratio, numk):
    # obtain data
	data = readfile('C:/Users/Administrator/Desktop/ML/project/caltech101/ImageProcessing/baseline-feature.mat')
	data_2 = drawdata(data['x'], data['y'], ratio, ordered=False)
	x_train = data_2['x_train'][:,:6000]
	x_test = data_2['x_test'][:,:6000]
	y_train = data_2['y_train']
	y_test = data_2['y_test']

	result = nnsvm_train(x_train, y_train, x_test, y_test, numk)

	return result

if __name__ == '__main__':
    result = training(0.8, 3)
    print(result)
