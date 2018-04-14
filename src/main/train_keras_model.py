 
from data_reading import imageList_generator
from data_reading import batch_generator as bg
import data_paths
import pickle
import numpy as np

import keras
from keras.metrics import top_k_categorical_accuracy

from keras_models import vgg_like_model

if __name__ == '__main__':
    samples = np.load(data_paths.train_samples)
    with open(data_paths.train_samples_species_map, 'rb') as f:
        species_map = pickle.load(f)

    top3_acc = functools.partial(top_k_categorical_accuracy, k=3)
    top3_acc.__name__ = 'top3_acc'
    top50_acc = functools.partial(top_k_categorical_accuracy, k=50)
    top50_acc.__name__ = 'top50_acc'

    model = vgg_like_model.get_model(species_map.keys())

    model.compile(optimizer=sgd_optimizer, loss='mse', metrics=['accuracy', top3_acc, top50_acc])

    for epoch in range(1):
        for x, y in bg.getNextImageBatch(samples, species_map):
            model.train_on_batch(x, y)



