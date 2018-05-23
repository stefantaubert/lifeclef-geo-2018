import settings_main as settings
import numpy as np
np.random.seed(settings.seed)
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, AveragePooling2D, MaxPooling2D
from keras import regularizers
import tensorflow as tf

def get_model(output_shape, channel_count):
    model = Sequential()
    l2_rate = 0.01
    activation = 'tanh'
    model.add(AveragePooling2D(input_shape=(channel_count, 64, 64), pool_size=(64, 64), data_format='channels_first'))
    model.add(BatchNormalization(axis=1))
    model.add(Flatten())
    model.add(Dropout(0.25))
    model.add(Dense(units=5000, activation=activation))
    model.add(BatchNormalization(axis=1))
    model.add(Dropout(0.25))
    model.add(Dense(units=5000, activation=activation))
    model.add(BatchNormalization(axis=1))
    model.add(Dropout(0.25))
    model.add(Dense(units=output_shape, activation='sigmoid'))
    return model