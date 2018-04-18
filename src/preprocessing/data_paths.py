import settings

def get_suffix_p():
    return "_p" + str(settings.pixel_count)

def get_suffix_pr():
    return get_suffix_p() + "_r" + str(settings.round_data_ndigits)

def get_suffix_pro():
    return get_suffix_pr() + "_o" + str(settings.min_occurence)

def get_suffix_prot():
    return get_suffix_pro() + "_t" + str(settings.threshold)

# auf linux muss ein ../ extra davor
root_lines = open("../../config/data_root").read().split('\n')
root = root_lines[0]

patch_train = root + "patchTrain"
patch_test = root + "patchTest"

occurrences_train_gen = root + "occurrences_train_gen" + get_suffix_p() + ".csv"
occurrences_test_gen = root + "occurrences_test_gen" + get_suffix_p() + ".csv"

occurrences_train = root + "occurrences_train.csv"
occurrences_test = root + "occurrences_test.csv"

preprocessing_dir = root + "preprocessing/"
train = preprocessing_dir + "train" + get_suffix_pro() + ".csv"
test = preprocessing_dir + "test" + get_suffix_pr() + ".csv"

train_with_groups = preprocessing_dir + "train_with_groups" + get_suffix_prot() + ".csv"

most_common_values = preprocessing_dir + "most_common_values" + get_suffix_pro() + ".csv"
named_groups = preprocessing_dir + "named_groups" + get_suffix_prot() + ".npy"
groups = preprocessing_dir + "groups" + get_suffix_prot() + ".txt"
similar_species = preprocessing_dir + "similar_species" + get_suffix_pro() + ".npy"
channel_map_diff = preprocessing_dir + "channel_map_diff" + get_suffix_pro() + ".csv"