import scipy.io as sio
import collections
import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib.pyplot as plt

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

if __name__ == '__main__':
    dataset = readfile('baseline-feature.mat')
    d2=truncate(dataset['x'],2)
    class_num=dataset['class_num'][0]
    class_total=dataset['class_total']
    dot_color=np.floor(np.linspace(0,class_total,class_total*class_num))
    dot_size=0.5
    plt.figure()
    plt.scatter(d2[:,0],d2[:,1],dot_size,dot_color,cmap='hsv')
    plt.savefig('feature_2d',dpi=800)
