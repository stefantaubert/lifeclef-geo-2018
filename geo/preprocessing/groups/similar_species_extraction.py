import os
import pickle
from tqdm import tqdm

from geo.preprocessing.groups.species_difference_extraction import load_channel_map_diff
from geo.preprocessing.text_preprocessing import load_train
from geo.data_paths import similar_species
from geo.settings import threshold

def load_similar_species():
    assert os.path.exists(similar_species)
    with open(similar_species, 'rb') as f:
        return pickle.load(f)

def extract_similar_species():
    if not os.path.exists(similar_species):
        _create()
    else: 
        print("Similar species already exist.")
            
def get_similar_species_dict(species_diff_matrix_df):
    _, species, species_c = load_train()
    species = species
    species_count = species_c
    assert len(species_diff_matrix_df.index) == species_count
    assert len(species_diff_matrix_df.columns.values) == species_count
    array = species_diff_matrix_df.as_matrix()
    
    similar_species = {k: [] for k in species}

    for i in tqdm(range(species_count)):
        for j in range(species_count):
            is_current_species = j == i      

            if not is_current_species:
                distance = array[i][j]

                if distance <= threshold:
                    current_species = species[i]
                    other_species = species[j]
                    similar_species[current_species].append(other_species)
    
    return similar_species

def _create():
    print("Get similar species...")
    species_diff_matrix = load_channel_map_diff()
    similar_species_dict = get_similar_species_dict(species_diff_matrix)
    print(similar_species_dict)
    pickle.dump(similar_species_dict, open(similar_species, 'wb'))

if __name__ == "__main__":
    _create()