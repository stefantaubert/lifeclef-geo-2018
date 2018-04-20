import settings_analysis as settings

def get_suffix():
    return "_p" + str(settings.pixel_count) + "_r" + str(settings.round_data_ndigits) + "_o" + str(settings.min_occurence)

def get_suffix_groups():
    return "_p" + str(settings.pixel_count) + "_r" + str(settings.round_data_ndigits) + "_o" + str(settings.min_occurence) + "_t" + str(settings.threshold)

# auf linux muss ein ../ extra davor
root_lines = open("../../config/data_root").read().split('\n')
root = root_lines[0]

occurrences_train_gen = root + "occurrences_train_gen_p" + str(settings.pixel_count) + ".csv"
occurrences_test_gen = root + "occurrences_test_gen_p" + str(settings.pixel_count) + ".csv"

patch_train = root + "patchTrain"
heatmaps = root + "analysis/Heatmaps/"

occurrences_train = root + "occurrences_train.csv"
occurrences_test = root + "occurrences_test.csv"

preprocessing_dir = root + "preprocessing/"
most_common_values = preprocessing_dir + "most_common_values" + get_suffix() + ".csv"
named_groups = preprocessing_dir + "named_groups" + get_suffix_groups() + ".npy"
groups = preprocessing_dir + "groups" + get_suffix_groups() + ".txt"
similar_species = preprocessing_dir + "similar_species" + get_suffix() + ".npy"
channel_map_diff = preprocessing_dir + "channel_map_diff" + get_suffix() + ".csv"

most_common_values_diagram = root + "analysis/most_common_values" + get_suffix() + ".pdf"
species_occurences_per_value = root + "analysis/species_occurences_per_value" + get_suffix() + ".pdf"
species_value_occurences = root + "analysis/species_value_occurences" + get_suffix() + ".pdf"
values_occurences_train = root + "analysis/values_occurences_train" + get_suffix() + ".pdf"
values_occurences_test = root + "analysis/values_occurences_test" + get_suffix() + ".pdf"

group_network = root + "analysis/group_network" + get_suffix_groups() + ".pdf"
group_length_probabilities = root + "analysis/group_length_probabilities" + get_suffix_groups() + ".pdf"
species_occurences = root + "analysis/species_occurences.csv"
species_channel_map_dir = root + "analysis/channel_maps" + get_suffix() + "/"
value_occurences_species_dir = root + "analysis/value_occurences_species" + get_suffix() + "/"

