from geo.preprocessing.image_to_csv import extract_occurences_train
from geo.preprocessing.image_to_csv import extract_occurences_test
from geo.preprocessing.text_preprocessing import extract_test
from geo.preprocessing.text_preprocessing import extract_train
from geo.preprocessing.groups.most_common_value_extraction import extract_most_common_values
from geo.preprocessing.groups.species_difference_extraction import extract_channel_map_diff
from geo.preprocessing.groups.similar_species_extraction import extract_similar_species
from geo.preprocessing.groups.group_extraction import extract_named_groups
from geo.preprocessing.groups.group_preprocessing import extract_train_with_groups

def extract_groups():
    extract_occurences_train()
    extract_train()
    extract_most_common_values()
    extract_channel_map_diff()
    extract_similar_species()
    extract_named_groups()
    extract_train_with_groups()

def create_trainset():
    extract_occurences_train()
    extract_train()

def create_testset():
    extract_occurences_test()
    extract_test()

def create_datasets():
    create_trainset()
    create_testset()

if __name__ == "__main__":
    #create_datasets()
    extract_groups()