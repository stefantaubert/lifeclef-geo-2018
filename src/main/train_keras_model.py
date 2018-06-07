#Trains the Keras model
#loads the train samples and splits them into train and validation samples
#loads the species map
#compiles the model
#fits the model on the batch generator using the train samples and the species map
#saves the model
import module_support_main
from data_reading import batch_generator as bg
import data_paths_main as data_paths
import pickle
import numpy as np
import keras
import metrics
import tensorflow as tf
from keras_models import vgg_like_model, global_average_model, denseNet
from keras.callbacks import ModelCheckpoint
import os
import settings_main as stg
from data_reading.imageList_generator import generate_train_image_list


def train_keras_model():
    if not os.path.exists(data_paths.train_samples):
        generate_train_image_list()

    np.random.seed(stg.seed)
    tf.reset_default_graph()
    a = tf.constant([1, 1, 1, 1, 1], dtype=tf.float32)
    graph_level_seed = 1
    operation_level_seed = 1
    tf.set_random_seed(graph_level_seed)
    b = tf.nn.dropout(a, 0.5, seed=operation_level_seed)

    samples = np.load(data_paths.train_samples)
    split = 1 - np.int(len(samples)*stg.train_val_split)
    samples, val_samples = samples[:split, :], samples[split:, :]

    with open(data_paths.train_samples_species_map, 'rb') as f:
        species_map = pickle.load(f)

    pickle.dump(species_map, open(data_paths.keras_training_species_map, 'wb'))

    top3_acc = metrics.get_top3_accuracy()
    top10_acc = metrics.get_top10_accuracy()
    top50_acc = metrics.get_top50_accuracy()

    #model = vgg_like_model.get_model(len(species_map.keys()), 33)
    #model = global_average_model.get_model(len(species_map.keys()), 33)

    model = denseNet.DenseNet(classes=len(species_map.keys()), nb_dense_block=3, nb_filter=32)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', top3_acc, top10_acc, top50_acc])

    checkpoint = ModelCheckpoint(data_paths.keras_training_model, monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only=True)

    print("start Training...")

    model.fit_generator(bg.nextBatch(samples, species_map, augment=False), epochs=stg.EPOCHS, steps_per_epoch=len(samples)/stg.BATCH_SIZE,
                        verbose=1, validation_data=bg.nextBatch(val_samples, species_map, augment=False), validation_steps=len(val_samples)/stg.BATCH_SIZE,
                        callbacks=[checkpoint])

    print("Finished!")

if __name__ == '__main__':
    train_keras_model()




    





