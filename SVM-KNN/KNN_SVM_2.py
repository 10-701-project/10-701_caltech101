'''
1. Find a collection of Ksl neighbors using a crude distance function (e.g. L2);

2. Compute the accurate distance function (e.g. tangent distance) on the Ksl samples and pick the K nearest neighbors;

3. Compute (or read from cache if possible) the pairwise accurate distance of the union of the K neighbors and the query;

4. Convert the pairwise distance matrix into a kernel matrix using the kernel trick;

5. Apply DAGSVM on the kernel matrix and label the query using the resulting classifier.
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

def nnsvm_train_2(x_train, y_train, x_test, y_test, k1, k2, d='l2'):
    '''
        p : integer, optional (default = 2)
        Power parameter for the Minkowski metric. When p = 1, this is
        equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.

        
    '''

    # initialization
    length_train = len(y_train)
    length_test = len(y_test)
    y_hat_train = np.zeros_like(y_train)
    y_hat_test = np.zeros_like(y_test)

    # training
    start = time.clock()
    knn = neighbors.NearestNeighbors(n_neighbors=k1, algorithm='auto', p=2)
    knn.fit(x_train)
    time_train = time.clock() - start

    # knn_train
    start = time.clock()

    nb = knn.kneighbors(x_train, return_distance=False)

    # Naive SVM_KNN_train
    knn_2 = neighbors.NearestNeighbors(n_neighbors=k2, algorithm='auto', metric=d)
    clf = SVC(kernel='rbf', gamma='auto')
    for i in range(length_train):
        knn_2.fit(x_train[nb[i]])
        nb_2 = knn_2.kneighbors(x_train[i].reshape(1, -1), return_distance=False)
        if k2 >= 2:
            x_temp = x_train[nb[i][nb_2[0]]]
            y_temp = y_train[nb[i][nb_2[0]]]
            if len(set(y_temp)) > 1:
                clf.fit(x_temp, y_temp)
                y_hat_train[i] = clf.predict(x_train[i].reshape(1, -1))[0]
            else:
                y_hat_train[i] = y_train[nb[i][nb_2[0]][0]]
        else:
            y_hat_train[i] = y_train[nb[i][nb_2[0]]]
        '''
        base = y_train[nb[i][0]]
        if all(y_train[j] == base for j in nb[i]):
            y_hat_train[i] = base
        else:
            x_temp = x_train[nb[i]]
            y_temp = y_train[nb[i]]
            clf = LinearSVC()
            clf.fit(x_temp, y_temp)
            y_hat_train[i] = clf.predict(x_train[i].reshape(1, -1))[0]
        '''

    time_test_train = time.clock() - start

    # result_train
    train_error = float((y_hat_train != y_train).mean())

    # knn_test
    start = time.clock()
    nb = knn.kneighbors(x_test, return_distance=False)

    # Naive SVM_KNN_test
    for i in range(length_test):
        knn_2.fit(x_train[nb[i]])
        nb_2 = knn_2.kneighbors(x_test[i].reshape(1, -1), return_distance=False)
        if k2 >= 2:
            x_temp = x_train[nb[i][nb_2[0]]]
            y_temp = y_train[nb[i][nb_2[0]]]
            if len(set(y_temp)) > 1:
                clf.fit(x_temp, y_temp)
                y_hat_test[i] = clf.predict(x_test[i].reshape(1, -1))[0]
            else:
                y_hat_test[i] = y_train[nb[i][nb_2[0]][0]]
        else:
            y_hat_test[i] = y_train[nb[i][nb_2[0]]]
    '''
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
    '''

    time_test_test = time.clock() - start

    # result_test
    test_error = float((y_hat_test != y_test).mean())

    time_train_avg = time_train / len(y_train)
    time_test_train_avg = time_test_train / len(y_train)
    time_test_test_avg = time_test_test / len(y_test)

    result = {
        'K1': k1,
        'K2': k2,
        'train_error': train_error,
        'test_error': test_error,
        'time_train_avg': time_train_avg,
        'time_test_train_avg': time_test_train_avg,
        'time_test_test_avg': time_test_test_avg,
    }
    return result


def training(ratio, distance, k1, k2):
    # obtain data
    # data = readfile(
        # 'C:/Users/Administrator/Desktop/ML/project/caltech101/ImageProcessing/baseline-feature.mat')
    data = readfile(
        'baseline-feature.mat')
    data_2 = drawdata(data['x'], data['y'], ratio, ordered=False)
    x_train = data_2['x_train'][:, :6000]
    x_test = data_2['x_test'][:, :6000]
    y_train = data_2['y_train']
    y_test = data_2['y_test']

    result = nnsvm_train_2(x_train, y_train, x_test, y_test, k1, k2, distance)

    return result


if __name__ == '__main__':
    result = dict()
    # f = open('result.txt', 'w')
    minimum = 1.0
    algo = ''

    distance_option = ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan', 'braycurtis', 'canberra', 'chebyshev', 'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']
    # distance_option = ['rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']

    for d in distance_option:
        f = open('result_'+d+'.txt', 'w')
        for i in range(20):
            result[i+1] = training(0.8, d, (i+1)*2, i+1)
            
            f.write('K1: '+ str(result[i+1]['K1'])+ '\n')
            f.write('K2: '+ str(result[i+1]['K2'])+ '\n')
            f.write('train_error: '+ str(result[i+1]['train_error'])+ '\n')
            f.write('test_error: '+ str(result[i+1]['test_error'])+ '\n')
            f.write('time_train_avg: '+ str(result[i+1]['time_train_avg'])+ '\n')
            f.write('time_test_train_avg: '+ str(result[i+1]['time_test_train_avg'])+ '\n')
            f.write('time_test_test_avg: '+ str(result[i+1]['time_test_test_avg'])+ '\n')

            f.write('-'*50+'\n')
            f.write('-'*50+'\n')

            if result[i+1]['test_error'] < minimum:
                minimum = result[i+1]['test_error']
                algo = d

            print(d, i, 'completed')
        # print('K1: ', result[i+1]['K1'])
        # print('K2: ', result[i+1]['K2'])
        # print('train_error: ', result[i+1]['train_error'])
        # print('test_error: ', result[i+1]['test_error'])
        # print('time_train_avg: ', result[i+1]['time_train_avg'])
        # print('time_test_train_avg: ', result[i+1]['time_test_train_avg'])
        # print('time_test_test_avg: ', result[i+1]['time_test_test_avg'])

        # print('-'*50)
        # print('-'*50)

        f.close()
        print(d + ' comppleted')

    print(minimum)
    print(algo)
