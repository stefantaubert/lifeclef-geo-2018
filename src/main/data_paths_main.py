from data_paths_global import *
import settings_main
import os 

test_submission = root + "submissions/submission_val.csv"

train_samples = root + "ImageLists/Train/samples.npy"
train_samples_species_map = root + "ImageLists/Train/species_map.py"

test_samples = root + 'ImageLists/Test/samples.npy'

#xgb normal training paths
xgb_dir = root + "xgb/"
xgb_group_map = xgb_dir + "group_map.npy"
xgb_species_map = xgb_dir + "species_map.npy"
xgb_prediction = xgb_dir + "prediction.npy"
xgb_glc_ids = xgb_dir + "glc_ids.npy"
xgb_submission = xgb_dir + "submission.csv"
xgb_dump = xgb_dir + "dump"
xgb_model = xgb_dir + "model"
# xgb_species_occurences = root + "analysis/species_occurences.csv"

#keras single model training paths
keras_training_gt = root + "keras_training_results/gt.npy"
keras_training_results = root + "keras_training_results/results.npy"
keras_training_species_map = root + "keras_training_results/species_map.py"
keras_training_submission = root + "keras_training_results/submission.csv"
keras_training_glc_ids = root + "keras_training_results/glc_ids.npy"
keras_training_model = root + 'keras_training_results/model.h5'

#keras multi model training paths

keras_multi_model_training_gt = root + "keras_multi_model_training_results/gt.npy"
keras_multi_model_training_results = root + "keras_multi_model_training_results/results.npy"
keras_multi_model_training_species_map = root + "keras_multi_model_training_results/species_map.py"
keras_multi_model_training_submission = root + "keras_multi_model_training_results/submission.csv"
keras_multi_model_training_glc_ids = root + "keras_multi_model_training_results/glc_ids.npy"
keras_multi_model_training_model1 = root + 'keras_multi_model_training_results/model1.h5'
keras_multi_model_training_model2 = root + 'keras_multi_model_training_results/model2.h5'
keras_multi_model_training_model3 = root + 'keras_multi_model_training_results/model3.h5'

if not os.path.exists(xgb_dir):
    os.makedirs(xgb_dir)
