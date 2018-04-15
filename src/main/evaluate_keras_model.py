#evaluates the model saved by the current training
#loads the previously saved model and validation samples
#iterates over all validation samples and saves predictions
#creates a submission for the predictions and evaluates this submission


from data_reading import batch_generator as bg
import data_paths
import pickle
import numpy as np
from keras import load_model
import settings as stg

from submission_maker import make_submission_for_current_training
from evaluation import evaluate_current_training_results

if __name__ == '__main__':
    samples = np.load(data_paths.train_samples)
    split = np.int(len(samples)*stg.train_val_split)
    samples, val_samples = samples[:split, :], samples[:split, :]
    print(len(val_samples))
    with open(data_paths.train_samples_species_map, 'rb') as f:
        species_map = pickle.load(f)

    model = load_model(data_paths.current_training_model)

    ground_truth = []
    predictions = []

    for x,y, species_ids in bg.nextValidationBatch(val_samples, species_map):
        ground_truth.extend(species_ids)
        predictions.extend(model.predict_on_batch(x))

    ground_truth = np.array(ground_truth)
    predictions = np.array(predictions)

    np.save(data_paths.current_training_gt, ground_truth)
    np.save(data_paths.current_training_results, predictions)
    pickle.dump(species_map, open(data_paths.current_training_species_map, 'wb'))

    make_submission_for_current_training()
    evaluate_current_training_results