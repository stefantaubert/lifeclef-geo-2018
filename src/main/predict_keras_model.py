#DEPRECATED
from data_reading import batch_generator as bg
import data_paths_main as data_paths
import pickle
import numpy as np
import settings_main as stg
from keras_models import vgg_like_model, denseNet
import tensorflow as tf
from data_reading.imageList_generator import generate_test_image_list
from submission import make_submission_from_files
import os

def predict_keras_model():
    if not os.path.exists(data_paths.test_samples):
        generate_test_image_list()

    np.random.seed(stg.seed)
    tf.reset_default_graph()
    a = tf.constant([1, 1, 1, 1, 1], dtype=tf.float32)
    graph_level_seed = 1
    operation_level_seed = 1
    tf.set_random_seed(graph_level_seed)
    b = tf.nn.dropout(a, 0.5, seed=operation_level_seed)

    samples = np.load(data_paths.test_samples)

    with open(data_paths.keras_training_species_map, 'rb') as f:
        species_map = pickle.load(f)

    #model = vgg_like_model.get_model(len(species_map.keys()), 33)

    model = denseNet.DenseNet(classes=len(species_map.keys()), nb_dense_block=3, nb_filter=32)

    model.load_weights(data_paths.keras_training_model)

    model.compile(optimizer='adam', loss='categorical_crossentropy')

    predictions = []
    glc_ids = []

    for x, batch_glc_ids in bg.nextTestBatch(samples, species_map):
        predictions.extend(model.predict_on_batch(x))
        glc_ids.extend(batch_glc_ids)

    predictions = np.array(predictions)
    glc_ids = np.array(glc_ids)

    np.save(data_paths.keras_test_results, predictions)
    np.save(data_paths.keras_test_glc_ids, glc_ids)

    make_submission_from_files(data_paths.keras_training_species_map,
                               data_paths.keras_test_results,
                               data_paths.keras_test_glc_ids,
                               data_paths.keras_test_submission,
                               header=False)

if __name__ == '__main__':
    predict_keras_model()