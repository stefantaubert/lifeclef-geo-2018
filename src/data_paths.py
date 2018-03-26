root_lines = open("../config/data_root").read().split('\n')
root = root_lines[0]

patch_train = root + "patchTrain"
occurrences_train = root + "occurrences_train.csv"
features_train = root + "train_features.csv"
submission_val = root + "submission_val.csv"#

x_img = root + "Numpy_Files/x_img.npy"
x_text = root + "Numpy_Files/x_text.npy"
y = root + "Numpy_Files/y.npy"
species_map = root + "Numpy_Files/species_map.p"