# Simple CNN model for CIFAR-10
import numpy as np
import os
from cv2 import imread, resize, INTER_CUBIC
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.callbacks import ModelCheckpoint
K.set_image_dim_ordering('th')


def load_data(data_path, train_test_ratio=0.7):
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []

    cat_index = 0
    for category in os.listdir(data_path):
        data_cat_dir = "{}/{}".format(data_path, category)
        if ".DS_Store" in data_cat_dir: continue

        img_names = os.listdir(data_cat_dir)
        for i in range(len(img_names)):
            img_name = img_names[i]
            if ".DS_Store" == img_name:
                continue

            img_path = "{}/{}".format(data_cat_dir, img_name)
            img = imread(img_path)
            img = resize(img, (150, 150), INTER_CUBIC)
            img.resize((3, 150, 150))
            if i <= len(img_names) * train_test_ratio:
                train_data.append(img)
                train_labels.append(cat_index)
            else:
                test_data.append(img)
                test_labels.append(cat_index)
        cat_index += 1
    return ((np.array(train_data), np.array(train_labels)),
            (np.array(test_data), np.array(test_labels)))


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

data_base_dir = "/Users/yunjinwu/Projects/cnn_model/101_ObjectCategories"

# load data
# X_train: (50000, 3, 32, 32)
# y_train: (50000, 1)
# X_test: (10000, 3, 32, 32)
# y_test: (10000, 1)
(X_train, y_train), (X_test, y_test) = load_data(data_base_dir)
print("Successfully load data! "
      "X_train.shape{}, y_train.shape{}, X_test.shape{}, y_test.shape{}"
      .format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))

# normalize inputs from 0-255 to 0.0-1.0
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# Create the model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(3, 150, 150), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Compile model
epochs = 25
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())

# define the checkpoint
filepath = "model.h5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs,
          batch_size=64, callbacks=callbacks_list, shuffle=True)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

