from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, AveragePooling2D
from DataReader import read_data
import numpy as np
from keras.metrics import top_k_categorical_accuracy
import tensorflow as tf
import functools


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

def run_Model():
    species_map, x_img, x_text, y = read_data()
    species_count = y.shape[1]
    print(species_count)

    top3_acc = functools.partial(top_k_categorical_accuracy, k=3)
    top3_acc.__name__ = 'top3_acc'
    top50_acc = functools.partial(top_k_categorical_accuracy, k=50)
    top50_acc.__name__ = 'top50_acc'


    model = nn_Model(species_count)
    model.compile(optimizer='sgd',
                  loss='mse',
                  metrics=[top3_acc,  top50_acc, 'accuracy'])
    model.fit(x_text, y, epochs=2, batch_size=32, validation_split=0.2)

    result = model.predict(np.array(x_text[0:1]))

    print(result)

if __name__ == '__main__':
    run_Model()

