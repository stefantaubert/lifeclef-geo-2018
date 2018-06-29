#Makes a full run for the Keras Model, including training, evaluation and prediction on the Test set


import pickle
import numpy as np
import keras
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
import os
import sys
from tqdm import tqdm

from geo.data_reading.batch_generator import nextBatch, nextValidationBatch, nextTestBatch
from geo.data_paths import train_samples_path, test_samples_path, train_samples_species_map
from geo.models.data_paths import keras_training_model, keras_test_submission
from geo.preprocessing.imagelist_generator import generate_train_image_list ,generate_test_image_list
from geo.models.settings import seed, train_val_split, KERAS_MODEL, CHANNEL_COUNT, AUGMENT, EPOCHS, BATCH_SIZE, TOP_N_SUBMISSION_RANKS
from geo.models.keras.model_architectures import vgg_like_model, dense_net
from geo.metrics.keras_metrics import get_top3_accuracy, get_top10_accuracy, get_top50_accuracy
from geo.metrics.mrr import mrr_score
from geo.postprocessing.submission_maker import make_submission_df
from geo.postprocessing.get_ranks import get_ranks_df


def train():
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', get_top3_accuracy(), get_top10_accuracy(), get_top50_accuracy()])

    checkpoint = ModelCheckpoint(keras_training_model, monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only=True)

    print("Start training...")

    model.fit_generator(nextBatch(train_samples, species_map, augment=AUGMENT), epochs=EPOCHS, steps_per_epoch=len(train_samples)/BATCH_SIZE/500,
                        verbose=1, validation_data=nextBatch(val_samples, species_map, augment=False), validation_steps=len(val_samples)/BATCH_SIZE/100,
                        callbacks=[checkpoint])

    print("Training finished!")

def evaluate():
    model.load_weights(keras_training_model)
    ground_truth = []
    predictions = []
    glc_ids = []

    print("Predict validation set...")
    for x, y, batch_species_ids, batch_glc_ids in nextValidationBatch(val_samples, species_map):
        ground_truth.extend(batch_species_ids)
        predictions.extend(model.predict_on_batch(x))
        glc_ids.extend(batch_glc_ids)

    ground_truth = np.array(ground_truth)
    predictions = np.array(predictions)
    glc_ids = np.array(glc_ids)

    print("Make submission...")
    evaluation_df = make_submission_df(TOP_N_SUBMISSION_RANKS, species_map, predictions, glc_ids)

    print("Evaluate submission...")
    ranks = get_ranks_df(evaluation_df, ground_truth, TOP_N_SUBMISSION_RANKS)
    score = mrr_score(ranks)
    print("MRR-Score:", score * 100,"%")

def predict():
    model.load_weights(keras_training_model)
    predictions = []
    glc_ids = []

    print("Predict test set...")
    for x, batch_glc_ids in nextTestBatch(test_samples, species_map):
        predictions.extend(model.predict_on_batch(x))
        glc_ids.extend(batch_glc_ids)

    predictions = np.array(predictions)
    glc_ids = np.array(glc_ids)

    print("Make submission...")
    prediction_df = make_submission_df(TOP_N_SUBMISSION_RANKS, species_map, predictions, glc_ids)
    print("Save submission...")
    prediction_df.to_csv(keras_test_submission, index=False, sep=";", header=None)
    

if __name__ == '__main__':
    if not os.path.exists(train_samples_path):
        generate_train_image_list()
    if not os.path.exists(test_samples_path):
        generate_test_image_list()

    np.random.seed(seed)
    tf.set_random_seed(seed)

    train_samples = np.load(train_samples_path)
    split = 1 - np.int(len(train_samples)*train_val_split)
    train_samples, val_samples = train_samples[:split, :], train_samples[split:, :]
    test_samples = np.load(test_samples_path)

    with open(train_samples_species_map, 'rb') as f:
        species_map = pickle.load(f)

    if KERAS_MODEL == 'VGG_like':
        model = vgg_like_model.get_model(len(species_map.keys()), CHANNEL_COUNT)
    elif KERAS_MODEL == 'DenseNet':
        model = dense_net.DenseNet(classes=len(species_map.keys()), nb_dense_block=3, nb_filter=32)
    else:
        print("The Model you specified in settings_main.py is not available")
        sys.exit()

    train()
    evaluate()
    predict()

      
