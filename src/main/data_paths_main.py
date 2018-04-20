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
xgb_train = root + "preprocessing/train_p32_r0_o20.csv"
xgb_train_groups = root + "preprocessing/train_with_groups_p32_r0_o20_t20.csv"
xgb_named_groups = root + "preprocessing/named_groups_p32_r0_o20_t20.npy"
xgb_species_occurences = root + "analysis/species_occurences.csv"

current_training_gt = root + "Current_Training_Results/gt.npy"
current_training_results = root + "Current_Training_Results/results.npy"
current_training_species_map = root + "Current_Training_Results/species_map.py"
current_training_submission = root + "Current_Training_Results/submission.csv"
current_training_glc_ids = root + "Current_Training_Results/glc_ids.npy"


current_training_model = root + 'Current_Training_Results/model.h5'