import settings
import numpy as np
np.random.seed(settings.seed)

import submission_maker
import evaluation
import DataReader
import settings
import time

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, AveragePooling2D

from keras import optimizers

import data_paths
import pickle
from scipy.stats import rankdata
import pandas as pd
from keras.metrics import top_k_categorical_accuracy
import tensorflow as tf
import functools
from sklearn.model_selection import train_test_split


def nn_Model(output_shape):
    model = Sequential()
    model.add(Dense(input_shape=(33, ), units=5000, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(input_shape=(33, ), units=5000, activation='relu'))
    model.add(Dropout(rate=0.25))
    model.add(Dense(input_shape=(33, ), units=5000, activation='relu'))
    model.add(Dropout(rate=0.25))
    model.add(Dense(output_shape, activation='sigmoid'))
    return model

def cnn_Model_2(output_shape):
    model = Sequential()
    model.add(Conv2D(input_shape=(33, 64, 64),
                     filters=64,
                     activation='relu',
                     kernel_size=(7, 7),
                     strides=(2, 2),
                     data_format='channels_first'))
    model.add(Conv2D(filters=64,
                     activation='relu',
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     padding='same',
                     data_format='channels_first'))
    model.add(Conv2D(filters=64,
                     activation='relu',
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     data_format='channels_first'))
    model.add(Dropout(rate=0.5))
    model.add(AveragePooling2D(pool_size=(4,4),
                               data_format='channels_first'))
    model.add(Flatten())
    model.add(Dense(units=output_shape, activation='softmax'))
    return model

def cnn_Model(output_shape):
    model = Sequential()
    model.add(Conv2D(input_shape=(33, 64, 64),
                     filters=64,
                     activation='relu',
                     kernel_size=(10, 10),
                     strides=(2, 2),
                     data_format='channels_first'))
    model.add(AveragePooling2D(pool_size=(9,9),
                               data_format='channels_first'))
    model.add(Flatten())
    model.add(Dense(units=output_shape, activation='sigmoid'))
    return model

def run_Model():
    print("Run model...")
    x_text = np.load(data_paths.x_text)
    x_img = np.load(data_paths.x_img)
    y = np.load(data_paths.y_array)
    
    with open(data_paths.species_map, 'rb') as f:
        species_map = pickle.load(f)

    classes_ = list(species_map.keys())
    np.save(data_paths.species_map_training, classes_)

    x_train = x_img
    species_count = y.shape[1]
    #print(species_count)

    top3_acc = functools.partial(top_k_categorical_accuracy, k=3)
    top3_acc.__name__ = 'top3_acc'
    top50_acc = functools.partial(top_k_categorical_accuracy, k=50)
    top50_acc.__name__ = 'top50_acc'
    
    sgd_optimizer = optimizers.SGD(lr=0.0000001, momentum=0.0, decay=0.0, nesterov=False)
    #adam nehmen

    #model = nn_Model(species_count)
    #model = cnn_Model(species_count)
    model = cnn_Model_2(species_count)
    model.compile(optimizer=sgd_optimizer,
                  loss='mse',
                  metrics=['accuracy', top3_acc, top50_acc])

    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y, test_size=settings.train_val_split, random_state=settings.seed)

    model.fit(x_train, y_train, epochs=3, batch_size=256, verbose=2)

    #result = model.predict(np.array(x_text[0:3]))
    result = model.predict(x_valid)

    np.save(data_paths.prediction, result)

if __name__ == '__main__':
    start_time = time.time()

    #DataReader.read_and_write_data()
    run_Model()
    submission_maker.make_submission()
    evaluation.evaluate_with_mrr()

    print("Total duration:", time.time() - start_time, "s")

