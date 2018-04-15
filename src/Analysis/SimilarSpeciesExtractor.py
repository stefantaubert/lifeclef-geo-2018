import pandas as pd
import numpy as np
import data_paths
import settings
from Data import Data
from tqdm import tqdm
from collections import Counter
import os
import pickle
import SpeciesDiffExtractor

def load():
    assert os.path.exists(data_paths.similar_species)

    with open(data_paths.similar_species, 'rb') as f:
        return pickle.load(f)

class SimilarSpeciesExtractor():
    def __init__(self):
        self.data = Data()
        self.data.load_train()
    
    def get_similar_species_dict(self, species_diff_matrix_df):
        assert len(species_diff_matrix_df.index) == self.data.species_count
        assert len(species_diff_matrix_df.columns.values) == self.data.species_count
        array = species_diff_matrix_df.as_matrix()
        
        similar_species = {k: [] for k in self.data.species}

        for i in tqdm(range(self.data.species_count)):
            for j in range(self.data.species_count):
                is_current_species = j == i      

                if not is_current_species:
                    distance = array[i][j]

                    if distance <= settings.threshold:
                        current_species = i + 1
                        other_species = j + 1
                        similar_species[current_species].append(other_species)
        
        return similar_species

    def _create(self):
        print("Get similar species...")
        species_diff_matrix = SpeciesDiffExtractor.load()
        similar_species_dict = self.get_similar_species_dict(species_diff_matrix)
        pickle.dump(similar_species_dict, open(data_paths.similar_species, 'wb'))


if __name__ == "__main__":
    SimilarSpeciesExtractor()._create()