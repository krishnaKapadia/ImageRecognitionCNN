#!/usr/bin/env python

"""Description:
The train.py is to build your CNN model, train the model, and save it for later evaluation(marking)
This is just a simple template, you feel free to change it according to your own style.
However, you must make sure:
1. Your own model is saved to the directory "model" and named as "model.h5"
2. The "test.py" must work properly with your model, this will be used by tutors for marking.
3. If you have added any extra pre-processing steps, please make sure you also implement them in "test.py" so that they can later be applied to test images.

©2018 Created by Yiming Peng and Bing Xue
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K
from parser import __loader__

import numpy as np
import tensorflow as tf
import random

# Set random seeds to ensure the reproducible results
from tensorflow.python.keras.layers import Conv3D, MaxPooling3D, Convolution2D, Activation, MaxPooling2D, Flatten, \
    Dropout

from test import load_images, convert_img_to_array

SEED = 309
np.random.seed(SEED)
random.seed(SEED)
tf.set_random_seed(SEED)

input_img_height = 300
input_img_width = 300

# training_cherry = __loader__('data/Train_data/cherry')

def construct_model():
    """
    Construct the CNN model.
    ***
        Please add your model implementation here, and don't forget compile the model
        E.g., model.compile(loss='categorical_crossentropy',
                            optimizer='sgd',
                            metrics=['accuracy'])
        NOTE, You must include 'accuracy' in as one of your metrics, which will be used for marking later.
    ***
    :return: model: the initial CNN model
    """
    model = Sequential()

    # Convolution Filters
    # model.add(Dense(units=64, activation='relu', input_dim=100))
    # model.add(Dense(units=10, activation='softmax'))
    # model.add(Conv3D(30, 10, input_shape=(input_img_width, input_img_height, 3)))
    # model.add(MaxPooling3D(pool_size=(2, 2, 2)))

    model.add(Convolution2D(32, 3, 3, input_shape=(input_img_width, input_img_height, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Dropout layer o stop over-fitting
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='sparse_categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
    return model


def train_model(model):
    """
    Train the CNN model
    ***
        Please add your training implementation here, including pre-processing and training
    ***
    :param model: the initial CNN model
    :return:model:   the trained CNN model
    """
    # Add your code here

    # Pre-processing

    model.fit(train_imgs, train_labels, epochs=20, batch_size=128)

    return model


def save_model(model):
    """
    Save the keras model for later evaluation
    :param model: the trained CNN model
    :return:
    """
    # ***
    #   Please remove the comment to enable model save.
    #   However, it will overwrite the baseline model we provided.
    # ***
    # model.save("model/model.h5")
    print("Model Saved Successfully.")


if __name__ == '__main__':

    # Load in data
    train_imgs, train_labels = load_images('./data/Train_data/cherry')
    # Convert images to numpy arrays (images are normalized with constant 255.0), and binarize categorical labels
    train_imgs, train_labels = convert_img_to_array(train_imgs, train_labels)
    # print(train_labels)
    # print(train_imgs)

    model = construct_model()
    model = train_model(model)
    save_model(model)
