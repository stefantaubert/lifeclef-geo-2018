src_dir = "D:/dev/Python/life-clef-geo-2018/code/src"
import sys  
sys.path.append("D:/dev/Python/life-clef-geo-2018/code/src/preprocessing")

import settings_main

root_lines = open("../config/data_root").read().split('\n')
root = root_lines[0]

patch_train = root + "patchTrain"
patch_test = root + "patchTest"

occurrences_train = root + "occurrences_train.csv"
occurrences_test = root + "occurrences_test.csv"
occurrences_train_gen = root + "occurrences_train_gen_p" + str(settings_main.pixel_count) + ".csv"
occurrences_test_gen = root + "occurrences_test_gen_p" + str(settings_main.pixel_count) + ".csv"

submission_val = root + "submissions/submission_val.csv"

train_samples = root + "ImageLists/Train/samples.npy"
train_samples_species_map = root + "ImageLists/Train/species_map.py"

test_samples = root + 'ImageLists/Test/samples.npy'

xgb_dir = root + "xgb/"
xgb_group_map = xgb_dir + "group_map.npy"
xgb_species_map = xgb_dir + "species_map.npy"
xgb_prediction = xgb_dir + "prediction.npy"
xgb_glc_ids = xgb_dir + "glc_ids.npy"
xgb_submission = xgb_dir + "submission.csv"
xgb_train = root + "preprocessing/train_p32_r0_o10.csv"
xgb_train_groups = root + "preprocessing/train_with_groups_p32_r0_o10_t20.csv"
xgb_named_groups = root + "preprocessing/named_groups_p32_r0_o10_t20.npy"
xgb_species_occurences = root + "analysis/species_occurences.csv"

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