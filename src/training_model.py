# -*- coding: utf-8 -*-
# @File    : training_model.py
# @Author  : Robin Lan
# @Time    : 7/3/23 19:37
# @Software: PyCharm
# @Description: this file is used to train the model and save the model to the disk for later use.


from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from matplotlib import pyplot as plt
from keras.datasets import mnist


def training_model():
    # Load MNIST data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Plot the first image in the dataset
    # print(X_train.shape, y_train.shape)
    # plt.imshow(X_train[0])
    # plt.show()

    # Reshape the data to fit the model
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
    # print(X_train.shape)

    # Convert data type to float32
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    # print(y_train.shape)

    # Convert 1-dimensional class arrays to 10-dimensional class matrices
    Y_train = np_utils.to_categorical(y_train, 10)
    Y_test = np_utils.to_categorical(y_test, 10)

    # create model
    model = Sequential()
    # add model layers
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # add dropout layer
    model.add(Dropout(0.25))

    # add flatten layer
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    # compile model using accuracy to measure model performance
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # train the model
    model.fit(X_train, Y_train, batch_size=32, epochs=10, verbose=1)
    model.save('models/model.h5')

    # # evaluate the model
    # score = model.evaluate(X_test, Y_test, verbose=0)
    # print('Test loss:', score[0])
