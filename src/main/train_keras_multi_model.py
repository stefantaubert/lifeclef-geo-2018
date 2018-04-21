#Trains the Keras model
#loads the train samples and splits them into train and validation samples
#loads the species map
#compiles the model
#fits the model on the batch generator using the train samples and the species map
#saves the model

from data_reading import single_channel_batch_generator as scbg
import data_paths_main as data_paths
import pickle
import numpy as np
import keras
import metrics
import tensorflow as tf
from keras_models import vgg_like_model
from keras.callbacks import ModelCheckpoint

import settings_main as stg

if __name__ == '__main__':
    tf.reset_default_graph()
    a = tf.constant([1, 1, 1, 1, 1], dtype=tf.float32)
    graph_level_seed = 1
    operation_level_seed = 1
    tf.set_random_seed(graph_level_seed)
    b = tf.nn.dropout(a, 0.5, seed=operation_level_seed)

    samples = np.load(data_paths.train_samples)
    split = np.int(len(samples)*stg.train_val_split)
    samples, val_samples = samples[:split, :], samples[:split, :]
    print(len(val_samples))
    with open(data_paths.train_samples_species_map, 'rb') as f:
        species_map = pickle.load(f)

    pickle.dump(species_map, open(data_paths.keras_multi_model_training_species_map, 'wb'))

    top3_acc = metrics.get_top3_accuracy()
    top50_acc = metrics.get_top50_accuracy()
    
    #train model 1
    model = vgg_like_model.get_model(len(species_map.keys()), 1)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', top3_acc, top50_acc])

    checkpoint = ModelCheckpoint(data_paths.keras_multi_model_training_model1, monitor='val_top3_acc', verbose=1, save_best_only=True, mode='max')

    model.fit_generator(scbg.nextBatch(samples, species_map, 31), epochs=stg.EPOCHS, steps_per_epoch=len(samples)/stg.BATCH_SIZE,
                        verbose=2, validation_data=scbg.nextBatch(val_samples, species_map, 31), validation_steps=len(val_samples)/stg.BATCH_SIZE,
                        callbacks=[checkpoint])

    #train model 2
    model = vgg_like_model.get_model(len(species_map.keys()), 1)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', top3_acc, top50_acc])

    checkpoint = ModelCheckpoint(data_paths.keras_multi_model_training_model2, monitor='val_top3_acc', verbose=1, save_best_only=True, mode='max')

    model.fit_generator(scbg.nextBatch(samples, species_map, 32), epochs=stg.EPOCHS, steps_per_epoch=len(samples)/stg.BATCH_SIZE,
                        verbose=2, validation_data=scbg.nextBatch(val_samples, species_map, 32), validation_steps=len(val_samples)/stg.BATCH_SIZE,
                        callbacks=[checkpoint])

    #train model 3
    model = vgg_like_model.get_model(len(species_map.keys()), 1)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', top3_acc, top50_acc])

    checkpoint = ModelCheckpoint(data_paths.keras_multi_model_training_model3, monitor='val_top3_acc', verbose=1, save_best_only=True, mode='max')

    model.fit_generator(scbg.nextBatch(samples, species_map, 27), epochs=stg.EPOCHS, steps_per_epoch=len(samples)/stg.BATCH_SIZE,
                        verbose=2, validation_data=scbg.nextBatch(val_samples, species_map, 27), validation_steps=len(val_samples)/stg.BATCH_SIZE,
                        callbacks=[checkpoint])
