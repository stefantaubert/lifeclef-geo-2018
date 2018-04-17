import pandas as pd
import numpy as np
import data_paths
from tqdm import tqdm
from collections import Counter
import os
import math
import MostCommonValueExtractor
import TextPreprocessing

def load():
    assert os.path.exists(data_paths.channel_map_diff)
    
    return pd.read_csv(data_paths.channel_map_diff)

def extract():
    if not os.path.exists(data_paths.similar_species):
        SpeciesDiffExtractor()._create()
    else: 
        print("Species distances already exist.")
            
#                 tmp = species_distances
#                 tmp.sort()
#                 print(tmp[:5])
#                 print(tmp[-5:])
#                 x = species
#                 y = species_distances

#                 plt.figure()
#                 plt.bar(x,y,align='center') # A bar chart
#                 plt.xlabel("species")
#                 plt.ylabel('dist')
#                 plt.show()
#                 #plt.savefig(data_paths.species_channel_map_dir + str(specie) + ".png", bbox_inches='tight')

class SpeciesDiffExtractor():
    def __init__(self):
        _, species, species_c = TextPreprocessing.load_train()
        self.species = species
        self.species_count = species_c
    
    def get_vector_length(self, v):
        summ = 0

        for num in v:
            summ += num * num

        distance = math.sqrt(summ)

        return distance

    def get_species_diff_matrix_df(self, most_common_value_matrix_df):
        most_common_value_matrix_df.drop(['occurence', 'species_glc_id'], axis=1, inplace=True)
        array = most_common_value_matrix_df.as_matrix()
        assert len(array) == self.species_count
        
        matrix = []

        for i in tqdm(range(self.species_count)):
            current_channel_map = np.array(array[i])
            species_distances = []

            for j in range(self.species_count):
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
        result_ser = pd.DataFrame(results_array, columns=self.species)

        return result_ser

    def _create(self):
        print("Calculate species distances...")
        most_common_value_matrix = MostCommonValueExtractor.load()

        species_diff_matrix = self.get_species_diff_matrix_df(most_common_value_matrix)
        species_diff_matrix.to_csv(data_paths.channel_map_diff, index=False)

if __name__ == "__main__":
    extract()