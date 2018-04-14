import settings

root_lines = open("../config/data_root").read().split('\n')
root = root_lines[0]

patch_train = root + "patchTrain"
patch_test = root + "patchTest"
occurrences_train = root + "occurrences_train.csv"
occurrences_test = root + "occurrences_test.csv"
occurrences_train_gen = root + "occurrences_train_gen_" + str(settings.pixel_count) + ".csv"
occurrences_test_gen = root + "occurrences_test_gen_" + str(settings.pixel_count) + ".csv"
features_train = root + "train_features.csv"
max_values_species = root + "max_values_species.csv"
species_channel_map_dir = root + "channel_maps/"
channel_map_diff = root + "channel_map_diff.csv"

model_dump = root + "xgboost.dmp"
model = root + "xgboost"

submission_val = root + "submission_val.csv"#
prediction = root + "prediction.npy"#

similar_species = root + "similar_species.npy"
x_img = root + "Numpy_Files/x_img.npy"
x_text = root + "Numpy_Files/x_text.npy"
y_array = root + "Numpy_Files/y.npy"
y_ids = root + "Numpy_Files/y_ids.npy"
species_map = root + "Numpy_Files/species_map.p"
species_map_training = root + "Numpy_Files/species_map.npy"

heatmaps = root + "Analysis/Heatmaps/"

image_train_set_y = root + "Numpy_Files/ImageFileList/train_y.npy"
image_train_set_x = root + "Numpy_Files/ImageFileList/train_x.npy"


train_samples = root + "Numpy_Files/Train/samples.npy"