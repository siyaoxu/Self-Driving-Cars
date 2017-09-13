
# coding: utf-8

import csv
import cv2
import h5py
import pickle
import numpy as np
from random import shuffle
from util import *
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
# get_ipython().magic('matplotlib inline')


def generator(lines, batch_size):
    '''
    lines should start from the second row of driving_log.csv
    '''
    steering_correction = 0.2
    while True:
        # shuffle the image list
        shuffle(lines)
        batch = lines[0:batch_size]

        # Create empty lists to contain batch of images and steering measurement
        batch_images = []
        batch_measurements = []
        # Loop and load this batch
        for line in batch:
            # randomly choose an image from center, left, right for augmentation
            choice = np.random.choice(3)
            if choice == 0:
                current_path = line[0]
                # the original center image
                image = procImg(cv2.imread(current_path))
                measurement = float(line[3])
            elif choice == 1:
                # left image
                current_path = line[1]
                image =  procImg(cv2.imread(current_path))
                measurement = float(line[3]) + steering_correction
            else:
                # right image
                current_path = line[2]
                image =  procImg(cv2.imread(current_path))
                measurement = float(line[3]) - steering_correction
            # randomly determine if we are going to augment the data
#            if np.random.rand() < 1.1:
            aug_image, aug_measurement = augmImg(image,measurement)
            batch_images.append(aug_image)
            batch_measurements.append(aug_measurement)
#            else:
#                batch_images.append(image)
#                batch_measurements.append(measurement)
        X_batch = np.array(batch_images)
        y_batch = np.array(batch_measurements)
        yield X_batch, y_batch


# define the model
def modelDef(input_shape, dropout = 0.6):
    model = Sequential()

    # lambda layer
    model.add(Lambda(lambda x: (x/255.0)-0.5,input_shape = input_shape))

    # Convolution layer 1 input 64x200x3 output 64x200x24
    model.add(Convolution2D(filters = 24, kernel_size = 5, activation = 'elu', padding = 'same'))
    # Maxpooling layer 1 input 64x200x24 ouput 32x100x24
    model.add(MaxPooling2D())
#     # Dropout layer 1
#     model.add(Dropout(rate = dropout))

    # Convolution layer 2 input 32x100x24 output 32x100x36
    model.add(Convolution2D(filters = 36, kernel_size = 5, activation = 'elu', padding = 'same'))
    # Maxpooling layer 2 input 32x100x36 ouput 16x50x36
    model.add(MaxPooling2D())
#     # Dropout layer 2
#     model.add(Dropout(rate = dropout))

    # Convolution layer 3 input 16x50x36 output 16x50x48
    model.add(Convolution2D(filters = 48, kernel_size = 5, activation = 'elu', padding = 'same'))
    # Maxpooling layer 3 input 16x50x32 ouput 8x25x32
    model.add(MaxPooling2D())
#    # Dropout layer 3
#    model.add(Dropout(rate = dropout))

    # Convolution layer 4 input 8x25x48 output 6x25x64
    model.add(Convolution2D(filters = 64, kernel_size = 3, activation = 'elu', padding = 'same'))

    # Convolution layer 5 input 6x23x64 output 4x25x64
    model.add(Convolution2D(filters = 64, kernel_size = 3, activation = 'elu', padding = 'same'))

    # Flatten the feature map
    model.add(Flatten())

#    # Fully connected layer 1 input 1024
#    model.add(Dense(1024, activation = 'relu'))
    # Dropout layer
    model.add(Dropout(rate = dropout))

    # Fully connected layer 2 input 100
    model.add(Dense(1000, activation = 'elu'))
#     # Dropout layer
#     model.add(Dropout(rate = dropout))

    # Fully connected layer 3 input 50
    model.add(Dense(500, activation = 'elu'))
#     # Dropout layer
#     model.add(Dropout(rate = dropout))

    # Fully connected layer 4 input 10
    model.add(Dense(100, activation = 'elu'))
#     # Dropout layer
#     model.add(Dropout(rate = dropout))

    # prediction
    model.add(Dense(1))
    return model



def trainModel(model, lines, batch_size, steps_per_epoch, learning_rate, validation_ratio = 0.25, epochs = 10,
               loss='mse',  modelName = 'test_model.h5', verbose = 1):
    optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss = loss, optimizer = optimizer)
    tr_generator = generator(lines, batch_size)
    v_generator = generator(lines, batch_size)
    validation_steps = np.int32(np.float32(steps_per_epoch)*validation_ratio)
    # checkpoint
    filepath = modelName
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    
    history = model.fit_generator(generator = tr_generator,
                        validation_data = v_generator,
                        steps_per_epoch = steps_per_epoch,
                        validation_steps = validation_steps,
                        epochs=epochs,
                        callbacks=callbacks_list,
                        verbose=verbose)
    return history


def readDataList(lines, paths):
    for path in paths:
        with open(path) as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                lines.append(line)
    return lines

if __name__ == "__main__":

    # load data
    lines = []
    paths = [
          '../../track1/driving_log.csv',
#          '../../track2/driving_log.csv',
#          '../../trach2_b/driving_log.csv',
    ]

    lines = readDataList(lines, paths)


    # set values for tuning parameters
    input_shape = (64, 200, 3)
    dropout = 0.5
    batch_size = 128
    steps_per_epoch = 1.6e2 # np.int32(np.float32(len(lines)*3)/np.float32(batch_size))
    learning_rate = 5e-4
    validation_ratio = 0.25
    epochs = 25

    # print(steps_per_epoch)

    # define and train the model
    model = modelDef(input_shape = input_shape, dropout = dropout)
    history = trainModel(model = model,
               lines = lines,
               batch_size = batch_size,
               steps_per_epoch = steps_per_epoch,
               learning_rate = learning_rate,
               epochs = epochs,
               validation_ratio = validation_ratio,
               loss='mse', modelName = 'model.h5', verbose =1 )

    with open('model_history', 'wb') as file_pi:
               pickle.dump(history.history, file_pi)

