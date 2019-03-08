import scipy.io as sio
import numpy as np
import os
from typing import Tuple, Any
import cv2


data_base_dir = "/Users/YWU/Projects/pytorch_test/101_ObjectCategories"
resized_data_base_dir = "/Users/YWU/Projects/pytorch_test/101_resized_data"


def resize_and_write_data(data_base_dir, resized_data_base_dir):
    i = 0
    for category in os.listdir(data_base_dir):
        data_cat_dir = "{}/{}".format(data_base_dir, category)
        data_write_dir = "{}/{}".format(resized_data_base_dir, category)

        try:
            os.stat(data_write_dir)
        except:
            os.mkdir(data_write_dir)

        if ".DS_Store" in data_cat_dir: continue
        for img_name in os.listdir(data_cat_dir):
            if ".DS_Store" == img_name: continue

            img_path = "{}/{}".format(data_cat_dir, img_name)
            dst_path = "{}/{}".format(data_write_dir, img_name)
            img = cv2.imread(img_path)
            i += 1
            print(i, img.shape)

            img_resized = cv2.resize(img, (150, 150), cv2.INTER_CUBIC)
            try:
                cv2.imwrite(dst_path, img_resized)
            except:
                print(img_path)


def load_data(data_path, train_test_ratio=0.7):
    train_data = []
    train_labels = []

    cat_index = 1
    for category in os.listdir(data_base_dir):
        data_cat_dir = "{}/{}".format(data_base_dir, category)
        if ".DS_Store" in data_cat_dir: continue
        for img_name in os.listdir(data_cat_dir):
            if ".DS_Store" == img_name: continue

            img_path = "{}/{}".format(data_cat_dir, img_name)
            img = cv2.imread(img_path)
            train_data.append(img)
            train_labels.append(cat_index)
        cat_index += 1

    slice_index = int(len(train_labels) * train_test_ratio)
    test_data = train_data[slice_index:]
    test_labels = train_labels[slice_index:]
    del train_data[slice_index:]
    del train_labels[slice_index:]
    return (train_data, train_labels), (test_data, test_labels)



