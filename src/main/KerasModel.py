#Makes a full run for the Keras Model, including training, evaluation and prediction on the Test set


import module_support_main
from data_reading import batch_generator as bg
import data_paths_main as data_paths
import pickle
import numpy as np
import keras
from metrics import get_top3_accuracy, get_top10_accuracy, get_top50_accuracy
import tensorflow as tf
from keras_models import vgg_like_model, global_average_model, denseNet
from keras.callbacks import ModelCheckpoint
import os
import sys
import settings_main as stg
from data_reading.imageList_generator import generate_train_image_list, generate_test_image_list
from submission import make_submission_from_files
from evaluation import evaluate_results_from_files


def train():
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', get_top3_accuracy(), get_top10_accuracy(), get_top50_accuracy()])

    checkpoint = ModelCheckpoint(data_paths.keras_training_model, monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only=True)

    print("Start training...")

    model.fit_generator(bg.nextBatch(train_samples, species_map, augment=stg.AUGMENT), epochs=stg.EPOCHS, steps_per_epoch=len(train_samples)/stg.BATCH_SIZE/1500,
                        verbose=1, validation_data=bg.nextBatch(val_samples, species_map, augment=False), validation_steps=len(val_samples)/stg.BATCH_SIZE,
                        callbacks=[checkpoint])

    print("Training finished!")

def evaluate():
    model.load_weights(data_paths.keras_training_model)
    ground_truth = []
    predictions = []
    glc_ids = []

    for x, y, batch_species_ids, batch_glc_ids in bg.nextValidationBatch(val_samples, species_map):
        print(model.predict_on_batch(x))
        ground_truth.extend(batch_species_ids)
        predictions.extend(model.predict_on_batch(x))
        glc_ids.extend(batch_glc_ids)

    ground_truth = np.array(ground_truth)
    predictions = np.array(predictions)
    glc_ids = np.array(glc_ids)

    np.save(data_paths.keras_training_gt, ground_truth)
    np.save(data_paths.keras_training_results, predictions)
    np.save(data_paths.keras_training_glc_ids, glc_ids)

    make_submission_from_files(species_map_path=data_paths.train_samples_species_map,
                               predictions_path=data_paths.keras_training_results,
                               glc_ids_path=data_paths.keras_training_glc_ids,
                               submission_path=data_paths.keras_training_submission)
                               
    evaluate_results_from_files(submission_path=data_paths.keras_training_submission,
                                gt_path=data_paths.keras_training_gt,
                                species_map_path=data_paths.train_samples_species_map)


def predict():
    model.load_weights(data_paths.keras_training_model)
    predictions = []
    glc_ids = []

    for x, batch_glc_ids in bg.nextTestBatch(test_samples, species_map):
        predictions.extend(model.predict_on_batch(x))
        glc_ids.extend(batch_glc_ids)

    predictions = np.array(predictions)
    glc_ids = np.array(glc_ids)

    np.save(data_paths.keras_test_results, predictions)
    np.save(data_paths.keras_test_glc_ids, glc_ids)

    make_submission_from_files(data_paths.train_samples_species_map,
                               data_paths.keras_test_results,
                               data_paths.keras_test_glc_ids,
                               data_paths.keras_test_submission,
                               header=False)
    

if __name__ == '__main__':
    if not os.path.exists(data_paths.train_samples):
        generate_train_image_list()
    if not os.path.exists(data_paths.test_samples):
        generate_test_image_list()

    np.random.seed(stg.seed)
    tf.set_random_seed(stg.seed)

    train_samples = np.load(data_paths.train_samples)
    split = 1 - np.int(len(train_samples)*stg.train_val_split)
    train_samples, val_samples = train_samples[:split, :], train_samples[split:, :]
    test_samples = np.load(data_paths.test_samples)

    with open(data_paths.train_samples_species_map, 'rb') as f:
        species_map = pickle.load(f)

    if stg.KERAS_MODEL == 'VGG_like':
        model = vgg_like_model.get_model(len(species_map.keys()), stg.CHANNEL_COUNT)
    elif stg.KERAS_MODEL == 'DenseNet':
        denseNet.DenseNet(classes=len(species_map.keys()), nb_dense_block=3, nb_filter=32)
    else:
        print("The Model you specified in settings_main.py is not available")
        sys.exit()

    train()
    evaluate()
    predict()

      
