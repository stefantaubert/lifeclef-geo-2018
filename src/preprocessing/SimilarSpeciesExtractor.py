import pandas as pd
import numpy as np
import data_paths_pre as data_paths
import settings_preprocessing
from tqdm import tqdm
from collections import Counter
import os
import pickle
import SpeciesDiffExtractor
import TextPreprocessing

def load():
    assert os.path.exists(data_paths.similar_species)

    with open(data_paths.similar_species, 'rb') as f:
        return pickle.load(f)

def extract():
    if not os.path.exists(data_paths.similar_species):
        SimilarSpeciesExtractor()._create()
    else: 
        print("Similar species already exist.")
            

class SimilarSpeciesExtractor():
    def __init__(self):
        _, species, species_c = TextPreprocessing.load_train()
        self.species = species
        self.species_count = species_c
    
    def get_similar_species_dict(self, species_diff_matrix_df):
        assert len(species_diff_matrix_df.index) == self.species_count
        assert len(species_diff_matrix_df.columns.values) == self.species_count
        array = species_diff_matrix_df.as_matrix()
        
        similar_species = {k: [] for k in self.species}

        for i in tqdm(range(self.species_count)):
            for j in range(self.species_count):
                is_current_species = j == i      

                if not is_current_species:
                    distance = array[i][j]

                    if distance <= settings_preprocessing.threshold:
                        current_species = self.species[i]
                        other_species = self.species[j]
                        similar_species[current_species].append(other_species)
        
        return similar_species

    def _create(self):
        print("Get similar species...")
        species_diff_matrix = SpeciesDiffExtractor.load()
        similar_species_dict = self.get_similar_species_dict(species_diff_matrix)
        pickle.dump(similar_species_dict, open(data_paths.similar_species, 'wb'))

if __name__ == "__main__":
    extract()