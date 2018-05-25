import global_settings as settings
import os,sys,inspect

def get_suffix_o():
    return "_o" + str(settings.min_occurence)

def get_suffix_p():
    return "_p" + str(settings.pixel_count)

def get_suffix_pr():
    return get_suffix_p() + "_r" + str(settings.round_data_ndigits)

def get_suffix_pro():
    return get_suffix_pr() + "_o" + str(settings.min_occurence)

# def get_suffix_prom():
#     return get_suffix_pro() + "_m" + str(settings.use_mean)

def get_suffix_prot():
    return get_suffix_pro() + "_t" + str(settings.threshold)

def get_suffix_prote():
    return get_suffix_prot() + "_e" + str(settings.min_edge_count)

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
config_path = parentdir + "/config/data_root"


root_lines = open(config_path).read().split('\n')
root = root_lines[0]

log = root + "log.txt"

patch_train = root + "patchTrain"
patch_test = root + "patchTest"

occurrences_train_gen = root + "occurrences_train_gen" + get_suffix_p() + ".csv"
occurrences_test_gen = root + "occurrences_test_gen" + get_suffix_p() + ".csv"

occurrences_train = root + "occurrences_train.csv"
occurrences_test = root + "occurrences_test.csv"

# Preprocessing
preprocessing_dir = root + "preprocessing/"
train = preprocessing_dir + "train" + get_suffix_pro() + ".csv"
test = preprocessing_dir + "test" + get_suffix_pr() + ".csv"
train_with_groups = preprocessing_dir + "train_with_groups" + get_suffix_prote() + ".csv"
most_common_values = preprocessing_dir + "most_common_values" + get_suffix_pro() + ".csv"
named_groups = preprocessing_dir + "named_groups" + get_suffix_prote() + ".npy"
similar_species = preprocessing_dir + "similar_species" + get_suffix_prot() + ".npy"
channel_map_diff = preprocessing_dir + "channel_map_diff" + get_suffix_pro() + ".csv"

if not os.path.exists(preprocessing_dir):
    os.makedirs(preprocessing_dir)

if not os.path.exists(log):
    file = open(log, 'w+')
    file.close()
