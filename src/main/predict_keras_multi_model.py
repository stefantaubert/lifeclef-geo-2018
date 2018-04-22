from data_reading import single_channel_batch_generator as scbg
import data_paths_main as data_paths
import pickle
import numpy as np
import settings_main as stg
from keras_models import vgg_like_model
import tensorflow as tf
from data_reading.imageList_generator import generate_test_image_list
from submission import make_submission_from_files
import os

if __name__ == '__main__':
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

    model1 = vgg_like_model.get_model(len(species_map.keys()), 1)
    model2 = vgg_like_model.get_model(len(species_map.keys()), 1)
    model3 = vgg_like_model.get_model(len(species_map.keys()), 1)

    model1.load_weights(data_paths.keras_multi_model_training_model1)
    model2.load_weights(data_paths.keras_multi_model_training_model2)
    model3.load_weights(data_paths.keras_multi_model_training_model3)

    model1.compile(optimizer='adam', loss='binary_crossentropy')
    model2.compile(optimizer='adam', loss='binary_crossentropy')
    model3.compile(optimizer='adam', loss='binary_crossentropy')

    model1_predictions = []
    model2_predictions = []
    model3_predictions = []

    glc_ids = []

    for x, batch_glc_ids in scbg.nextTestBatch(samples, species_map, 31):
        glc_ids.extend(batch_glc_ids)
        model1_predictions.extend(model1.predict_on_batch(x))

    for x, batch_glc_ids in scbg.nextTestBatch(samples, species_map, 32):
        model2_predictions.extend(model2.predict_on_batch(x))

    for x, batch_glc_ids in scbg.nextTestBatch(samples, species_map, 27):
        model3_predictions.extend(model3.predict_on_batch(x))

    model1_predictions = np.array(model1_predictions)
    model2_predictions = np.array(model2_predictions)
    model3_predictions = np.array(model3_predictions)

    predictions = (model1_predictions + model2_predictions + model3_predictions) / 3
    glc_ids = np.array(glc_ids)

    np.save(data_paths.keras_multi_model_test_results, predictions)
    np.save(data_paths.keras_multi_model_test_glc_ids, glc_ids)

    make_submission_from_files(species_map_path=data_paths.keras_multi_model_training_species_map,
                               predictions_path=data_paths.keras_multi_model_test_results,
                               glc_ids_path=data_paths.keras_multi_model_test_glc_ids,
                               submission_path=data_paths.keras_multi_model_test_submission)
