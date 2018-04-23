from data_paths_global import *
import settings_analysis as settings
import os

analysis_dir = root + "analysis/"

heatmaps = root + "analysis/Heatmaps/"

most_common_values_best_features = analysis_dir + "most_common_values_best_features" + get_suffix_pro() + ".txt"
most_common_values_unique_count = analysis_dir + "most_common_values_unique_count" + get_suffix_pro() + ".pdf"
most_common_values_diagram = analysis_dir + "most_common_values" + get_suffix_pro() + ".pdf"
species_occurences_per_value = analysis_dir + "species_occurences_per_value" + get_suffix_pro() + ".pdf"
species_value_occurences = analysis_dir + "species_value_occurences" + get_suffix_pro() + ".pdf"
values_occurences_train = analysis_dir + "values_occurences_train" + get_suffix_pro() + ".pdf"
values_occurences_test = analysis_dir + "values_occurences_test" + get_suffix_pro() + ".pdf"

group_network = analysis_dir + "group_network" + get_suffix_prot() + ".pdf"
group_length_probabilities = analysis_dir + "group_length_probabilities" + get_suffix_prot() + ".pdf"
species_occurences = analysis_dir + "species_occurences.csv"
species_channel_map_dir = analysis_dir + "channel_maps" + get_suffix_pro() + "/"
value_occurences_species_dir = analysis_dir + "value_occurences_species" + get_suffix_pro() + "/"


if not os.path.exists(analysis_dir):
    os.makedirs(analysis_dir)