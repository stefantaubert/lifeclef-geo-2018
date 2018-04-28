#evaluates the model saved by the current training
#loads the previously saved model and validation samples
#iterates over all validation samples and saves predictions
#creates a submission for the predictions and evaluates this submission


from data_reading import batch_generator as bg
import data_paths_main as data_paths
import pickle
import numpy as np
import settings_main as stg
from keras_models import vgg_like_model
from submission import make_submission_from_files
from evaluation import evaluate_results_from_files

def evaluate_keras_model():
    samples = np.load(data_paths.train_samples)
    split = 1 - np.int(len(samples)*stg.train_val_split)
    samples, val_samples = samples[:split, :], samples[split:, :]

    with open(data_paths.keras_training_species_map, 'rb') as f:
        species_map = pickle.load(f)

    model = vgg_like_model.get_model(len(species_map.keys()), 33)

    model.load_weights(data_paths.keras_training_model)

    model.compile(optimizer='adam', loss='categorical_crossentropy')

    ground_truth = []
    predictions = []
    glc_ids = []

    for x, y, batch_species_ids, batch_glc_ids in bg.nextValidationBatch(val_samples, species_map):
        ground_truth.extend(batch_species_ids)
        predictions.extend(model.predict_on_batch(x))
        glc_ids.extend(batch_glc_ids)

    ground_truth = np.array(ground_truth)
    predictions = np.array(predictions)
    glc_ids = np.array(glc_ids)

    np.save(data_paths.keras_training_gt, ground_truth)
    np.save(data_paths.keras_training_results, predictions)
    np.save(data_paths.keras_training_glc_ids, glc_ids)

    make_submission_from_files(species_map_path=data_paths.keras_training_species_map,
                               predictions_path=data_paths.keras_training_results,
                               glc_ids_path=data_paths.keras_training_glc_ids,
                               submission_path=data_paths.keras_training_submission)
                               
    evaluate_results_from_files(submission_path=data_paths.keras_training_submission,
                                gt_path=data_paths.keras_training_gt,
                                species_map_path=data_paths.keras_training_species_map)

if __name__ == '__main__':
    evaluate_keras_model()