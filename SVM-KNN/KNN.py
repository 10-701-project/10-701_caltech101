# -*- coding: utf-8 -*-

import collections
import datetime
import os
import random
import sys
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
# from sklearn import *
from sklearn import neighbors
from sklearn.externals import joblib
from sklearn.metrics import classification_report, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC

from dataprocessing import drawdata, readfile


def knn_train(x_train, y_train, x_test, y_test, numk):
	'''
	p : integer, optional (default = 2)
        Power parameter for the Minkowski metric. When p = 1, this is
        equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.
	'''
	clf = neighbors.classification.KNeighborsClassifier(n_neighbors=numk, algorithm='auto', p=2)
	
	start = time.clock()
	clf.fit(x_train, y_train)
	time_train = time.clock() - start
    
	start = time.clock()
	y_hat_train = clf.predict(x_train)
	time_test_train = time.clock() - start

	start = time.clock()
	y_hat_test = clf.predict(x_test)
	time_test_test = time.clock() - start

	train_error = float((y_hat_train != y_train).mean())
	test_error = float((y_hat_test != y_test).mean())

	time_train_avg = time_train / len(y_train)
	time_test_train_avg = time_test_train / len(y_train)
	time_test_test_avg = time_test_test / len(y_test)

	result = {
			'K': numk,
			'train_error': train_error,
			'test_error': test_error,
			'time_train_avg': time_train_avg,
			'time_test_train_avg': time_test_train_avg,
			'time_test_test_avg': time_test_test_avg,
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

	result = knn_train(x_train, y_train, x_test, y_test, numk)

	return result
	

if __name__ == '__main__':
	result = dict()
	for i in range(20):
		result[i+1] = training(0.8, i+1)
		for j in result[i+1].keys():
			print(j, result[i+1][j])
		print('-'*50)
		print('-'*50)