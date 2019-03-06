import scipy.io as sio
import collections
import numpy as np
import random
from datetime import datetime
from sklearn.decomposition import PCA, TruncatedSVD

def readfile(path):
    mat_contents = sio.loadmat(path)

    # total number of classes
    class_total = len(mat_contents['classes'][0])

    # name of classes in a dict
    class_name = [mat_contents['classes'][0][i][0] for i in range(class_total)]

    # num of figures in each class
    # array, shape:(total, )
    class_num = np.array([mat_contents['classes_num'][0][i][0,0] for i in range(class_total)])

    # total number of figures in dataset
    figure_total = np.sum(class_num)

    # y: label of each figure, shape:(figure_total, )
    y = mat_contents['imageClass'][0]
    # x: features of each figure, shape:(figure_total, num of features)
    x = mat_contents['psix'].T

    result = {
        'class_total': class_total,
        'figure_total': figure_total,
        'class_name': class_name,
        'class_num': class_num,
        'y': y,
        'x': x,
    }
    return result

def truncate(x, n):
    foo = TruncatedSVD(n_components=n)
    newData = foo.fit_transform(x)
    return newData

def drawdata(x, y, ratio, ordered = True):
    '''
    To randomly draw a same ratio of figures from each class to train dataset
    The left goes to test dataset
    '''

    # initialization
    y_train_no = []
    y_test_no = []
    random.seed(datetime.now())

    label = set(y)
    label_num = collections.Counter(y)
    label_dict = dict()

    for i in label:
        label_dict[i] = np.where(y == i)[0]
        j = int(round(ratio*len(label_dict[i])))
        temp = random.sample(list(label_dict[i]), j)

        y_train_no += temp
        y_test_no += list(set(label_dict[i]).difference(set(temp)))
    
    if ordered:
        y_train_no.sort()
        y_test_no.sort()
    else:
        random.shuffle(y_train_no)
        random.shuffle(y_test_no)

    x_train = x[y_train_no]
    x_test = x[y_test_no]
    y_test = y[y_test_no]
    y_train = y[y_train_no]

    result = {
        'x_train': x_train,
        'x_test': x_test,
        'y_train': y_train,
        'y_test': y_test,
    }
    return result

if __name__ == '__main__':
    dataset = readfile('C:/Users/Administrator/Desktop/ML/project/caltech101/ImageProcessing/baseline-feature.mat')
    y = np.array([1,1,1,2,2,2,3,3,3])
    x = np.array([[0,0,0],[1,1,1],[2,2,2],[3,3,3],[4,4,4],[5,5,5],[6,6,6],[7,7,7],[8,8,8]])
    print(drawdata(x, y, 0.5, ordered=False))
    print('completed')