import settings
import numpy as np
np.random.seed(settings.seed)



from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, AveragePooling2D



import tensorflow as tf


def get_model(output_shape):
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