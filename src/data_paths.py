root_lines = open("../config/data_root").read().split('\n')
root = root_lines[0]

patch_train = root + "patchTrain"
occurrences_train = root + "occurrences_train.csv"
occurrences_train_gen = root + "occurrences_train_gen.csv"
features_train = root + "train_features.csv"

model_dump = root + "dump.dmp"
model = root + "xgboost"

submission_val = root + "submission_val.csv"#
prediction = root + "prediction.npy"#

x_img = root + "Numpy_Files/x_img.npy"
x_text = root + "Numpy_Files/x_text.npy"
y_array = root + "Numpy_Files/y.npy"
y_ids = root + "Numpy_Files/y_ids.npy"
species_map = root + "Numpy_Files/species_map.p"
species_map_training = root + "Numpy_Files/species_map.npy"