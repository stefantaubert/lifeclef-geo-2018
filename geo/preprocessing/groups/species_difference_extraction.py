import pandas as pd
import numpy as np
import os
import math
from tqdm import tqdm
from collections import Counter

from geo.preprocessing.text_preprocessing import load_train
from geo.preprocessing.groups.most_common_value_extraction import load_most_common_values
from geo.data_paths import channel_map_diff
from geo.data_paths import similar_species

def load_channel_map_diff():
    assert os.path.exists(channel_map_diff)
    return pd.read_csv(channel_map_diff)

def extract_channel_map_diff():
    if not os.path.exists(similar_species):
        _create()
    else: 
        print("Species distances already exist.")
  
def get_vector_length(v):
    summ = 0

    for num in v:
        summ += num * num

    distance = math.sqrt(summ)

    return distance

def get_species_diff_matrix_df(most_common_value_matrix_df):
    _, species, species_count = load_train()
    most_common_value_matrix_df.drop(['occurence', 'species_glc_id'], axis=1, inplace=True)
    array = most_common_value_matrix_df.as_matrix()
    assert len(array) == species_count
    
    matrix = []

    for i in tqdm(range(species_count)):
        current_channel_map = np.array(array[i])
        species_distances = []

        for j in range(species_count):
            is_current_channel_map = j == i

            if is_current_channel_map:
                species_distances.append(0)
            else:
                other_channel_map = np.array(array[j])
                diff_vector = other_channel_map - current_channel_map
                distance = get_vector_length(diff_vector)
                species_distances.append(distance)

        matrix.append(species_distances)
    
    #list to array to add to the dataframe as a new column
    results_array = np.asarray(matrix) 
    result_ser = pd.DataFrame(results_array, columns=species)

    return result_ser

def _create():
    print("Calculate species distances...")
    most_common_value_matrix = load_most_common_values()

    species_diff_matrix = get_species_diff_matrix_df(most_common_value_matrix)
    species_diff_matrix.to_csv(channel_map_diff, index=False)

if __name__ == "__main__":
    _create()