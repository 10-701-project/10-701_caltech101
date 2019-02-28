# -*- coding: utf-8 -*-
'''
A naive version of the SVM-KNN is: for a query,

1. compute distances of the query to all training examples and pick the nearest K neighbors;

2. if the K neighbors have all the same labels, the query is labeled and exit; else, compute the pairwise distances between the K neighbors;

3. convert the distance matrix to a kernel matrix and apply multiclass SVM;

4. use the resulting classifier to label the query.
'''

import os
import sys
import numpy as np
from sklearn import neighbors
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import datetime
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
import time
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


def nnsvm_train(x_train, y_train, x_test, y_test, numk, rfile):
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
    distance, nb = knn.kneighbours(x_train)

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
            y_hat_train[i] = clf.predict(x_train[i])[0]

    # result_train
    train_error = float((y_hat_train != y_train).mean())
    

    # knn_test
    distance, nb = knn.kneighbours(x_test)

    # Naive SVM_KNN_test
    for i in range(length_test):
        base = y_test[nb[i][0]]
        if all(y_test[j] == base for j in nb[i]):
            y_hat_test[i] = base
        else:
            x_temp = x_train[nb[i]]
            y_temp = y_train[nb[i]]
            clf = LinearSVC()
            clf.fit(x_temp, y_temp)
            y_hat_test[i] = clf.predict(x_test[i])[0]
    
    # result_test
    test_error = float((y_hat_test != y_test).mean())

'''
def incre(x_train, y_train, x_test, y_test, x_incre, y_incre, numk, rfile):
    X = np.concatenate((x_train, x_incre))
    y = np.concatenate((y_train, y_incre))
    print("***********incre**********")
    time1 = datetime.datetime.now()
    nnsvm_train(X, y, x_test, y_test, numk, rfile)
    time2 = datetime.datetime.now()
    print(time2-time1)
    print("**************************")
'''

def training():
    #train_file = 'data/train_data.txt'
    #test_file = 'data/test_data.txt'
    '''
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    incre_file = sys.argv[3]
    numk = int(sys.argv[4])
    print(train_file)
    print(test_file)

    X_train, y_train = file2np(train_file)
    X_test, y_test = file2np(test_file)
    X_incre, y_incre = file2np(incre_file)
    print("***********raw**********")
    time1 = datetime.datetime.now()
    rfile = 'result/'+str(numk)+'_'+train_file+"_result"
    nnsvm_train(X_train, y_train, X_test, y_test, numk, rfile)
    time2 = datetime.datetime.now()
    print(time2-time1)
    print("************************")
    rfile = 'result/'+str(numk)+'_'+incre_file+"_result"
    incre(X_train, y_train, X_test, y_test, X_incre, y_incre, numk, rfile)
    print("********************\n\n")
    '''

if __name__ == '__main__':
    # training()
    samples = np.array([[0., 0., 0.], [0., .5, 0.], [1., 1., .5]])
    neigh = neighbors.NearestNeighbors(n_neighbors=2)
    neigh.fit(samples) # doctest: +ELLIPSIS
    # NearestNeighbors(algorithm='auto', leaf_size=30, ...)
    print(neigh.kneighbors([[1., 1., 1.],[2,2,2]])) # doctest: +ELLIPSIS
    print(samples[np.array([0,2])])
        
