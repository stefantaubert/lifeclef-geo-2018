from data_reading import single_channel_batch_generator as scbg
import data_paths_main as data_paths
import pickle
import numpy as np
import settings_main as stg
from keras_models import vgg_like_model


from submission_maker import make_submission_from_files
from evaluation import evaluate_results_from_files

if __name__ == '__main__':
    samples = np.load(data_paths.train_samples)
    split = np.int(len(samples)*stg.train_val_split)
    samples, val_samples = samples[:split, :], samples[:split, :]
    print(len(val_samples))
    with open(data_paths.keras_multi_model_training_species_map, 'rb') as f:
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
    
    ground_truth = []
    model1_predictions = []
    model2_predictions = []
    model3_predictions = []

    glc_ids = []

    for x, y, batch_species_ids, batch_glc_ids in scbg.nextValidationBatch(val_samples, species_map, 31):
        ground_truth.extend(batch_species_ids)
        glc_ids.extend(batch_glc_ids)
        model1_predictions.extend(model1.predict_on_batch(x))

    for x, y, batch_species_ids, batch_glc_ids in scbg.nextValidationBatch(val_samples, species_map, 32):
        model2_predictions.extend(model2.predict_on_batch(x))

    for x, y, batch_species_ids, batch_glc_ids in scbg.nextValidationBatch(val_samples, species_map, 27):
        model3_predictions.extend(model3.predict_on_batch(x))


    model1_predictions = np.array(model1_predictions)
    model2_predictions = np.array(model2_predictions)
    model3_predictions = np.array(model3_predictions)

    predictions = (model1_predictions + model2_predictions + model3_predictions) / 3
    ground_truth = np.array(ground_truth)
    glc_ids = np.array(glc_ids)

    np.save(data_paths.keras_multi_model_training_gt, ground_truth)
    np.save(data_paths.keras_multi_model_training_results, predictions)
    np.save(data_paths.keras_multi_model_training_glc_ids, glc_ids)

    make_submission_from_files(species_map_path=data_paths.keras_multi_model_training_species_map,
                               predictions_path=data_paths.keras_multi_model_training_results,
                               glc_ids_path=data_paths.keras_multi_model_training_glc_ids,
                               submission_path=data_paths.keras_multi_model_training_submission)
                               
    evaluate_results_from_files(submission_path=data_paths.keras_multi_model_training_submission,
                                gt_path=data_paths.keras_multi_model_training_gt,
                                species_map_path=data_paths.keras_multi_model_training_species_map)