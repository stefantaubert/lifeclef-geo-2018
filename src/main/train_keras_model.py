#Trains the Keras model
#loads the train samples and splits them into train and validation samples
#loads the species map
#compiles the model
#fits the model on the batch generator using the train samples and the species map
#saves the model

from data_reading import batch_generator as bg
import data_paths
import pickle
import numpy as np
import keras
import metrics

from keras_models import vgg_like_model

import settings as stg

if __name__ == '__main__':
    samples = np.load(data_paths.train_samples)
    split = np.int(len(samples)*stg.train_val_split)
    samples, val_samples = samples[:split, :], samples[:split, :]
    print(len(val_samples))
    with open(data_paths.train_samples_species_map, 'rb') as f:
        species_map = pickle.load(f)

    top3_acc = metrics.get_top3_accuracy()
    top50_acc = metrics.get_top50_accuracy()

    model = vgg_like_model.get_model(len(species_map.keys()), 33)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', top3_acc, top50_acc])

    print("start Training...")

    model.fit_generator(bg.nextBatch(samples, species_map), epochs=stg.EPOCHS, steps_per_epoch=len(samples)/stg.BATCH_SIZE, verbose=1)

    print("saving Model...")

    model.save_weights(data_paths.current_training_model)

    print("Finished!")




    





