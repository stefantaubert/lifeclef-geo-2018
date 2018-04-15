import pandas as pd
import numpy as np
import data_paths
from Data import Data
from tqdm import tqdm
from collections import Counter
import os
import math
import MostCommonValueExtractor

def load():
    assert os.path.exists(data_paths.channel_map_diff)
    
    return pd.read_csv(data_paths.channel_map_diff)

class SpeciesDiffExtractor():
    def __init__(self):
        self.data = Data()
        self.data.load_train()
    
    def get_vector_length(self, v):
        summ = 0

        for num in v:
            summ += num * num

        distance = math.sqrt(summ)

        return distance

    def get_species_diff_matrix_df(self, most_common_value_matrix_df):
        most_common_value_matrix_df.drop(['occurence', 'species_glc_id'], axis=1, inplace=True)
        array = most_common_value_matrix_df.as_matrix()
        assert len(array) == self.data.species_count
        
        matrix = []

        for i in tqdm(range(self.data.species_count)):
            current_channel_map = np.array(array[i])
            species_distances = []

            for j in range(self.data.species_count):
                is_current_channel_map = j == i

                if is_current_channel_map:
                    species_distances.append(0)
                else:
                    other_channel_map = np.array(array[j])
                    diff_vector = other_channel_map - current_channel_map
                    # betrag des Vektors ausrechnen
                    distance = self.get_vector_length(diff_vector)
                    species_distances.append(distance)

            matrix.append(species_distances)
        
        results_array = np.asarray(matrix) #list to array to add to the dataframe as a new column
        result_ser = pd.DataFrame(results_array, columns=self.data.species)

        return result_ser

    def _create(self):
        print("Calculate species distances...")
        most_common_value_matrix = MostCommonValueExtractor.load()

        species_diff_matrix = self.get_species_diff_matrix_df(most_common_value_matrix)
        species_diff_matrix.to_csv(data_paths.channel_map_diff, index=False)

if __name__ == "__main__":
    SpeciesDiffExtractor()._create()