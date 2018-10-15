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
from keras import optimizers
from keras.layers import ZeroPadding2D
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras import backend as K
from parser import __loader__

import matplotlib.pyplot as plt
import numpy as np

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


import keras as k
import tensorflow as tf
import random
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print(sess.run(c))

# Set random seeds to ensure the reproducible results
from keras.layers import Conv3D, MaxPooling3D, Convolution2D, Activation, MaxPooling2D, Flatten, \
    Dropout
from keras.optimizers import Adam

from test import load_images, convert_img_to_array

SEED = 309
np.random.seed(SEED)
random.seed(SEED)
tf.set_random_seed(SEED)

scaled_img_dimensions = (224, 224)

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
    # vgg = k.applications.vgg16.VGG16()
    # # vgg = k.applications.resnet50.ResNet50()
    # # print(type(vgg))
    # model = Sequential()
    # for layer in vgg.layers:
    #     print(type(layer))
    #     model.add(layer)
    # model.layers.pop()
    #
    # for layer in model.layers:
    #     layer.trainable = False
    #
    # model.add(Dense(3, activation='softmax'))

    # Convolution Filters
    # model.add(Dense(units=64, activation='relu', input_dim=100))
    # model.add(Dense(units=10, activation='softmax'))
    # model.add(Conv3D(30, 10, input_shape=(input_img_width, input_img_height, 3)))
    # model.add(MaxPooling3D(pool_size=(2, 2, 2)))

    # model.add(Convolution2D(32, 3, input_shape=(200, 200, 3)))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    # print('1')
    # model.add(Convolution2D(32, 3))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    # print('2')
    # model.add(Convolution2D(64, 3))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    # print('3')
    # # Dropout layer o stop over-fitting
    # model.add(Flatten())

    # model.add(Dense(1000, activation='relu'))
    # model.add(Dropout(0.5))
    #
    # model.add(Dense(1000, activation='relu'))
    # model.add(Dropout(0.5))
    #
    # model.add(Dense(3))
    # model.add(Activation('softmax'))
    # model.add(Dropout(0.5))
    # model.add(Dense(1))
    # model.add(Activation('sigmoid'))


    adam = Adam(lr=0.001)

    model = Sequential()

    # model.add(Convolution2D(64, 11, 4, activation='relu', input_shape=(224, 224, 3)))
    # model.add(MaxPooling2D(3, 2))
    # model.add(Convolution2D(192, 5, 1, activation='relu'))
    # model.add(MaxPooling2D(3, 2))
    # model.add(Convolution2D(384, (3, 3), strides=1, padding='same', activation='relu'))
    # model.add(Convolution2D(256, (3, 3), strides=1, padding='same', activation='relu'))
    # model.add(Convolution2D(256, (3, 3), strides=1, padding='same', activation='relu'))
    # model.add(MaxPooling2D(3, 2))
    # model.add(Flatten())
    # model.add(Dense(2048, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(2048, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(3, activation='relu'))

    model.add(Convolution2D(20, 5, 5, border_mode="same",input_shape=(224, 224, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # second set of CONV => RELU => POOL
    model.add(Convolution2D(50, 5, 5, border_mode="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # set of FC => RELU layers
    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation("relu"))

    # softmax classifier
    model.add(Dense(3))
    model.add(Activation("softmax"))

    model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])
    print(model.summary())
    return model


def preprocess_data(img):
    # All image pre-processing is done here
    img = k.preprocessing.image.array_to_img(img, scale=False)

    # Scale down to a smaller image
    img = img.resize(scaled_img_dimensions)

    img = k.preprocessing.image.img_to_array(img)
    return img


def train_model(model):
    """
    Train the CNN model
    ***
        Please add your training implementation here, including pre-processing and training
    ***
    :param model: the initial CNN model
    :return:model:   the trained CNN model
    """

    # Pre-processing
    print('preprocessing')
    # Scale the image down to 90x90 & Apply random transform,
    datagenerator = ImageDataGenerator(
        # featurewise_center=True, featurewise_std_normalization=True,
                                       rotation_range=360, shear_range=0.4,
                                       horizontal_flip=True, vertical_flip=True,
                                       zoom_range=0.4, rescale=0.4,
                                       preprocessing_function=preprocess_data
    )

    train_batches = datagenerator.flow_from_directory('data/Train_data', target_size=scaled_img_dimensions, classes=['cherry', 'strawberry', 'tomato'], batch_size=500)

    imgs, labels = next(train_batches)
    plot(imgs, titles=labels)

    # Training
    # model.fit(generated_imgs, train_labels[0], epochs=20, batch_size=2000)
    model.fit_generator(train_batches, shuffle=True, epochs=8, steps_per_epoch=35)

    # Testing
    test_batches = datagenerator.flow_from_directory('data/Test_data', target_size=scaled_img_dimensions, classes=['cherry', 'strawberry', 'tomato'], batch_size=15)

    test_imgs, test_labels = next(test_batches)
    plot(test_imgs, titles=test_labels)

    predictions = model.predict_generator(test_batches, steps=1)

    print(predictions)

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


# Display batch of images using pyplot
def plot(imgs, figsize=(12, 6), rows=1, interp=False, titles=None):
    if type(imgs[0]) is np.ndarray:
        imgs = np.array(imgs).astype(np.uint8)
        if imgs.shape[-1] != 3:
            imgs = imgs.transpose((0, 2, 3, 1))
    f = plt.figure(figsize=figsize)
    cols = len(imgs)//rows if len(imgs) % 2 == 0 else len(imgs)//rows + 1

    for i in range(len(imgs)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(imgs[i], interpolation=None if interp else 'none')
    plt.show()


if __name__ == '__main__':

    # Load in data
    # train_imgs, train_labels = load_images('./data/Train_data/cherry')

    # Convert images to numpy arrays (images are normalized with constant 255.0), and binarize categorical labels
    # train_imgs, train_labels = convert_img_to_array(train_imgs, train_labels)
    # print(train_labels)
    # print(train_imgs)

    model = construct_model()
    model = train_model(model)
    save_model(model)
