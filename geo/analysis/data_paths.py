import os

from geo.data_paths import get_suffix_pro
from geo.data_paths import get_suffix_prote
from geo.data_paths import get_suffix_o
from geo.data_dir_config import root

analysis_dir = root + "analysis/"
heatmaps = root + "analysis/Heatmaps/"

most_common_values_best_features = analysis_dir + "most_common_values_best_features" + get_suffix_pro() + ".txt"
most_common_values_unique_count = analysis_dir + "most_common_values_unique_count" + get_suffix_pro() + ".pdf"
most_common_values_diagram = analysis_dir + "most_common_values" + get_suffix_pro() + ".pdf"
species_occurences_per_value = analysis_dir + "species_occurences_per_value" + get_suffix_pro() + ".pdf"
species_value_occurences = analysis_dir + "species_value_occurences" + get_suffix_pro() + ".pdf"
values_occurences_train = analysis_dir + "values_occurences_train" + get_suffix_pro() + ".pdf"
values_occurences_test = analysis_dir + "values_occurences_test" + get_suffix_pro() + ".pdf"

group_network = analysis_dir + "group_network" + get_suffix_prote() + ".pdf"
group_length_probabilities = analysis_dir + "group_length_probabilities" + get_suffix_prote() + ".pdf"
species_channel_map_dir = analysis_dir + "channel_maps" + get_suffix_pro() + "/"
value_occurences_species_dir = analysis_dir + "value_occurences_species" + get_suffix_pro() + "/"

if not os.path.exists(analysis_dir):
    os.makedirs(analysis_dir)