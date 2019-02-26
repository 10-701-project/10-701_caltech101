# -*- coding: utf-8 -*-

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


def knn_train(x_train, y_train, x_test, y_test, numk):
	'''
	p : integer, optional (default = 2)
        Power parameter for the Minkowski metric. When p = 1, this is
        equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.
	'''
	clf = neighbors.KNeighborsClassifier(n_neighbors=numk, algorithm='auto', p=2)
	
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

'''
def incre(x_train, y_train, x_test, y_test, x_incre, y_incre, numk):
	X = np.concatenate((x_train, x_incre))
	y = np.concatenate((y_train, y_incre))
	print("***********incre**********")
	time1 = datetime.datetime.now()
	knn_train(X, y, x_test, y_test, numk)
	time2 = datetime.datetime.now()
	print(time2-time1)
	print("**************************")
'''

def training():
	# train_file = 'data/train_data.txt'
	# test_file = 'data/test_data.txt' 

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
	knn_train(X_train, y_train, X_test, y_test, numk)
	time2 = datetime.datetime.now()
	print(time2-time1 )
	print("************************")

	incre(X_train, y_train, X_test, y_test, X_incre, y_incre, numk)
	print("********************\n\n")
	'''

	
	

if __name__ == '__main__':
	training()


