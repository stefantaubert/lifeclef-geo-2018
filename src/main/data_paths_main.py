from data_paths_global import *
import settings_main
import os 



test_submission = root + "submissions/submission_val.csv"

img_list_dir = root + "image_lists/"
train_samples = img_list_dir + "train/samples.npy"
train_samples_species_map = img_list_dir + "train/species_map.py"
test_samples = img_list_dir + "test/samples.npy"

#xgb normal training paths
xgb_dir = root + "xgb/"
xgb_group_map = xgb_dir + "group_map.npy"
xgb_species_map = xgb_dir + "species_map.npy"
xgb_prediction = xgb_dir + "prediction.npy"
xgb_glc_ids = xgb_dir + "glc_ids.npy"
#xgb_trainchached = xgb_dir + "train.txt" muss libsvm format sein (phraug2 -> pivottable von csv)
xgb_submission = xgb_dir + "submission.csv"
xgb_test_prediction = xgb_dir + "test_prediction.npy"
xgb_test_glc_ids = xgb_dir + "test_glc_ids.npy"
xgb_test_submission = xgb_dir + "test_submission.csv"
xgb_model = xgb_dir + "model"
xgb_model_dump = xgb_dir + "model_dump"
xgb_feature_importances = xgb_dir + "feature_importances.pdf"
# xgb_species_occurences = root + "analysis/species_occurences.csv"

regression_dir = root + "regression/"
regression_species = regression_dir + "species.npy"
regression_prediction = regression_dir + "validation_prediction.npy"
regression_val_submission = regression_dir + "val_submission.csv"
regression_test_prediction = regression_dir + "test_prediction.npy"
regression_test_submission = regression_dir + "test_submission.csv"

#keras single model training paths
keras_training_dir = root + "keras_training_results/"
keras_training_gt = keras_training_dir + "gt.npy"
keras_training_results = keras_training_dir + "results.npy"
keras_training_species_map = keras_training_dir + "species_map.py"
keras_training_submission = keras_training_dir + "submission.csv"
keras_training_glc_ids = keras_training_dir + "glc_ids.npy"
keras_training_model = keras_training_dir + "model.h5"

#keras multi model training paths
keras_multi_model_training_dir = root + "keras_multi_model_training_results/"
keras_multi_model_training_gt = keras_multi_model_training_dir + "gt.npy"
keras_multi_model_training_results = keras_multi_model_training_dir + "results.npy"
keras_multi_model_training_species_map = keras_multi_model_training_dir + "species_map.py"
keras_multi_model_training_submission = keras_multi_model_training_dir + "submission.csv"
keras_multi_model_training_glc_ids = keras_multi_model_training_dir + "glc_ids.npy"
keras_multi_model_training_model1 = keras_multi_model_training_dir + "model1.h5"
keras_multi_model_training_model2 = keras_multi_model_training_dir + "model2.h5"
keras_multi_model_training_model3 = keras_multi_model_training_dir + "model3.h5"
keras_multi_model_training_model4 = keras_multi_model_training_dir + "model4.h5"
keras_multi_model_training_model5 = keras_multi_model_training_dir + "model5.h5"
keras_multi_model_training_model6 = keras_multi_model_training_dir + "model6.h5"

#keras single model test paths
keras_test_dir = root + "keras_predictions/"
keras_test_results = keras_test_dir + "results.npy"
keras_test_glc_ids = keras_test_dir + "glc_ids.npy"
keras_test_submission = keras_test_dir + "submission.csv"

#keras multi model test paths
keras_multi_model_test_dir = root + "keras_multi_model_predictions/"
keras_multi_model_test_results = keras_multi_model_test_dir + "results.npy"
keras_multi_model_test_glc_ids = keras_multi_model_test_dir + "glc_ids.npy"
keras_multi_model_test_submission = keras_multi_model_test_dir + "submission.csv"

if not os.path.exists(xgb_dir):
    os.makedirs(xgb_dir)

if not os.path.exists(regression_dir):
    os.makedirs(regression_dir)

if not os.path.exists(img_list_dir):
    os.makedirs(img_list_dir)
    os.makedirs(img_list_dir+"train/")
    os.makedirs(img_list_dir+"test/")

if not os.path.exists(keras_multi_model_training_dir):
    os.makedirs(keras_multi_model_training_dir)

if not os.path.exists(keras_training_dir):
    os.makedirs(keras_training_dir)

if not os.path.exists(keras_test_dir):
    os.makedirs(keras_test_dir)

if not os.path.exists(keras_multi_model_test_dir):
    os.makedirs(keras_multi_model_test_dir)
