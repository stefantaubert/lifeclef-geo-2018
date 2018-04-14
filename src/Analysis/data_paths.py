import settings

root_lines = open("../config/data_root").read().split('\n')
root = root_lines[0]

occurrences_train_gen = root + "occurrences_train_gen_" + str(settings.pixel_count) + ".csv"
occurrences_test_gen = root + "occurrences_test_gen_" + str(settings.pixel_count) + ".csv"

patch_train = root + "patchTrain"
heatmaps = root + "analysis/Heatmaps/"

occurrences_train = root + "occurrences_train.csv"
occurrences_test = root + "occurrences_test.csv"

max_values_species = root + "analysis/max_values_species.csv"
species_channel_map_dir = root + "analysis/channel_maps/"
channel_map_diff = root + "analysis/channel_map_diff.csv"
similar_species = root + "analysis/similar_species.npy"