#!/usr/bin/env python

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np
import keras as k
import tensorflow as tf
import random
# Set random seeds to ensure the reproducible results
from keras.layers import Conv3D, MaxPooling3D, Convolution2D, Activation, MaxPooling2D, Flatten, \
    Dropout
from keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix

from google.colab import drive

drive.mount('/content/drive')

import tensorflow as tf
print(tf.test.gpu_device_name())

SEED = 309
np.random.seed(SEED)
random.seed(SEED)
tf.set_random_seed(SEED)

scaled_img_dimensions = [64, 64]
epochs = 4

def construct_perceptron():
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(60, activation='relu'))
    model.add(Dense(3, activation='softmax'))
  
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
    adam = Adam(lr=0.001)

    model = Sequential()
    model.add(Convolution2D(16, 5, 5, activation='relu', input_shape=(scaled_img_dimensions[0], scaled_img_dimensions[1], 3)))
    model.add(MaxPooling2D(2, 2))

    model.add(Convolution2D(32, 5, 5, activation='relu'))
    model.add(MaxPooling2D(2, 2))

    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='binary_crossentropy',
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
                                       rotation_range=360, shear_range=0.6,
                                       horizontal_flip=True, vertical_flip=True,
                                       zoom_range=0.6, rescale=1./255,
                                       # preprocessing_function=preprocess_data
    )
    
    train_path = '/content/drive/My Drive/Colab Notebooks/Data/Train_data'
    test_path = '/content/drive/My Drive/Colab Notebooks/Data/Test_data'
    
    train_batches = datagenerator.flow_from_directory(train_path, target_size=scaled_img_dimensions, classes=['cherry', 'strawberry', 'tomato'], batch_size=32, class_mode='categorical')
    
    # Training
    # model.fit(generated_imgs, train_labels[0], epochs=20, batch_size=2000)
    model.fit_generator(train_batches, shuffle=True, epochs=2, steps_per_epoch=4500) 

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
    train_path = '/content/drive/My Drive/Colab Notebooks/Data/Train_data'
#     model.save("model/model.h5")
    model.save(train_path + "/model.h5")
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
