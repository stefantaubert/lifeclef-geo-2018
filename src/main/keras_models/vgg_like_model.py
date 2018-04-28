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
    l2_rate = 0.3
    activation = 'relu'
    model.add(Conv2D(input_shape=(channel_count, 64, 64), filters=32, activation='relu', kernel_size=(3, 3), strides=(1, 1), padding='same', data_format='channels_first', kernel_regularizer=regularizers.l2(l2_rate)))
    model.add(BatchNormalization(axis=1))
    model.add(Conv2D(filters=32, activation='relu', kernel_size=(3, 3), strides=(1, 1), padding='same', data_format='channels_first', kernel_regularizer=regularizers.l2(l2_rate)))
    model.add(BatchNormalization(axis=1))
    #model.add(Conv2D(filters=64, activation='relu', kernel_size=(3, 3), strides=(1, 1), padding='same', data_format='channels_first', kernel_regularizer=regularizers.l2(l2_rate)))
    #model.add(BatchNormalization(axis=1))
    #model.add(Conv2D(filters=64, activation='relu', kernel_size=(3, 3), strides=(1, 1), padding='same', data_format='channels_first', kernel_regularizer=regularizers.l2(l2_rate)))
    #model.add(BatchNormalization(axis=1))
    model.add(MaxPooling2D(pool_size=(2,2), data_format='channels_first'))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=64, activation='relu', kernel_size=(3, 3),  strides=(1, 1), padding='same', data_format='channels_first', kernel_regularizer=regularizers.l2(l2_rate)))
    model.add(BatchNormalization(axis=1))
    #model.add(Conv2D(filters=128, activation='relu', kernel_size=(3, 3),  strides=(1, 1), padding='same', data_format='channels_first', kernel_regularizer=regularizers.l2(l2_rate)))
    #model.add(BatchNormalization(axis=1))
    #model.add(Conv2D(filters=128, activation='relu', kernel_size=(3, 3),  strides=(1, 1), padding='same', data_format='channels_first', kernel_regularizer=regularizers.l2(l2_rate)))
    #model.add(BatchNormalization(axis=1))
    model.add(MaxPooling2D(pool_size=(2,2), data_format='channels_first'))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=64, activation='relu', kernel_size=(3, 3), strides=(1, 1), padding='same', data_format='channels_first', kernel_regularizer=regularizers.l2(l2_rate)))
    model.add(BatchNormalization(axis=1))
    #model.add(Conv2D(filters=256, activation='relu', kernel_size=(3, 3), strides=(1, 1), padding='same', data_format='channels_first', kernel_regularizer=regularizers.l2(l2_rate)))
    #model.add(BatchNormalization(axis=1))
    #model.add(Conv2D(filters=256, activation='relu', kernel_size=(3, 3), strides=(1, 1), padding='same', data_format='channels_first', kernel_regularizer=regularizers.l2(l2_rate)))
    #model.add(BatchNormalization(axis=1))
    model.add(MaxPooling2D(pool_size=(2,2), data_format='channels_first'))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=128, activation='relu', kernel_size=(3, 3), strides=(1, 1), padding='same', data_format='channels_first', kernel_regularizer=regularizers.l2(l2_rate)))
    model.add(BatchNormalization(axis=1))
    #model.add(Conv2D(filters=512, activation='relu', kernel_size=(3, 3), strides=(1, 1), padding='same', data_format='channels_first', kernel_regularizer=regularizers.l2(l2_rate)))
    #model.add(BatchNormalization(axis=1))
    #model.add(Conv2D(filters=512, activation='relu', kernel_size=(3, 3), strides=(1, 1), padding='same', data_format='channels_first', kernel_regularizer=regularizers.l2(l2_rate)))
    #model.add(BatchNormalization(axis=1))

    model.add(AveragePooling2D(pool_size=(8, 8), data_format='channels_first'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(units=output_shape, activation='sigmoid'))
    return model