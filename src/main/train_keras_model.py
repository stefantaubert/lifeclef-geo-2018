 
from data_reading import imageList_generator
from data_reading import batch_generator as bg
import data_paths
import pickle
import numpy as np
import functools
import keras
from keras.metrics import top_k_categorical_accuracy

from keras_models import vgg_like_model

if __name__ == '__main__':
    samples = np.load(data_paths.train_samples)
    split = np.int(len(samples)*0.8)
    samples, val_samples = samples[:split, :], samples[split:, :]
    with open(data_paths.train_samples_species_map, 'rb') as f:
        species_map = pickle.load(f)

    top3_acc = functools.partial(top_k_categorical_accuracy, k=3)
    top3_acc.__name__ = 'top3_acc'
    top50_acc = functools.partial(top_k_categorical_accuracy, k=50)
    top50_acc.__name__ = 'top50_acc'

    model = vgg_like_model.get_model(len(species_map.keys()))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', top3_acc, top50_acc])

    print("start Training")

    model.fit_generator(bg.getNextImageBatch(samples, species_map), epochs=1, steps_per_epoch=len(samples)/32)

    print(model.evaluate_generator(bg.getNextImageBatch(val_samples, species_map), steps=len(val_samples)/32))





