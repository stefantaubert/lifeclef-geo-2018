#reads the csv of the train andr test set and saves
#a list of patch_ids and patch_dirnames of all images in this set,
#and for the train set also
#a list of target vectors and the species map used to create the target vectors
import module_support
import numpy as np
import pandas as pd
from tqdm import tqdm
import tifffile
import pickle
import sys
import data_paths_main as data_paths


def generate_train_image_list():
    samples = []

    df = pd.read_csv(data_paths.occurrences_train, sep=';', low_memory=False)

    species_map = {l: i for i, l in enumerate(df.species_glc_id.unique())}

    print("Reading CSV...")

    for index, row in tqdm(df.iterrows(), miniters=100):
        current_species_glc_id = row["species_glc_id"]
        current_patch_dirname = row["patch_dirname"]
        current_patch_id = row["patch_id"]

        #Create Sample Entry, contains
        #Dirname of image patch, ID of image patch, species_id  
        sample =[current_patch_dirname, current_patch_id, current_species_glc_id]
        
        samples.append(sample)
    
    print("Writing Data...")

    samples = np.array(samples, dtype=np.uint32)

    np.save(data_paths.train_samples, samples)
    pickle.dump(species_map, open(data_paths.train_samples_species_map, 'wb'))


def generate_test_image_list():
    samples = []

    df = pd.read_csv(data_paths.occurrences_test, sep=';', low_memory=False)

    print("Reading CSV...")

    
    for index, row in tqdm(df.iterrows(), miniters=100):
        current_patch_dirname = row["patch_dirname"]
        current_patch_id = row["patch_id"]

        #Create Test Sample Entry, contains
        #Dirname of image patch, ID of image patch, species is set to 1 but will bei ignored later
        sample =[current_patch_dirname, current_patch_id, 1]
        
        samples.append(sample)
    
    print("Writing Data...")

    samples = np.array(samples, dtype=np.uint32)

    np.save(data_paths.test_samples, samples)

if __name__ == '__main__':
    #generate__train_image_list()
    generate_test_image_list()