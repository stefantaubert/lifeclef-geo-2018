import settings

# auf linux muss ein ../ extra davor
root_lines = open("../config/data_root").read().split('\n')
root = root_lines[0]

occurrences_train_gen = root + "occurrences_train_gen_" + str(settings.pixel_count) + ".csv"
occurrences_test_gen = root + "occurrences_test_gen_" + str(settings.pixel_count) + ".csv"

patch_train = root + "patchTrain"
heatmaps = root + "analysis/Heatmaps/"

occurrences_train = root + "occurrences_train.csv"
occurrences_test = root + "occurrences_test.csv"

most_common_values = root + "analysis/most_common_values_p" + str(settings.pixel_count) + "_r" + str(settings.round_data_ndigits) + ".csv"
most_common_values_diagram = root + "analysis/most_common_values_p" + str(settings.pixel_count) + "_r" + str(settings.round_data_ndigits) + ".pdf"
species_occurences_per_value = root + "analysis/species_occurences_per_value_p" + str(settings.pixel_count) + "_r" + str(settings.round_data_ndigits) + ".pdf"
species_value_occurences = root + "analysis/species_value_occurences_p" + str(settings.pixel_count) + "_r" + str(settings.round_data_ndigits) + ".pdf"
values_occurences_train = root + "analysis/values_occurences_train_p" + str(settings.pixel_count) + "_r" + str(settings.round_data_ndigits) + ".pdf"
values_occurences_test = root + "analysis/values_occurences_test_p" + str(settings.pixel_count) + "_r" + str(settings.round_data_ndigits) + ".pdf"

species_occurences = root + "analysis/species_occurences.csv"
species_channel_map_dir = root + "analysis/channel_maps_p" + str(settings.pixel_count) + "_r" + str(settings.round_data_ndigits) + "/"
value_occurences_species_dir = root + "analysis/value_occurences_species_p" + str(settings.pixel_count) + "_r" + str(settings.round_data_ndigits) + "/"
channel_map_diff = root + "analysis/channel_map_diff_p" + str(settings.pixel_count) + "_r" + str(settings.round_data_ndigits) + ".csv"
similar_species = root + "analysis/similar_species_p" + str(settings.pixel_count) + "_r" + str(settings.round_data_ndigits) + ".npy"