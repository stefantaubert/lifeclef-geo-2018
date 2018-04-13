#reads the csv of the train andr test set and saves
#a list of patch_ids and patch_dirnames of all images in this set,
#and for the train set also
#a list of target vectors and the species map used to create the target vectors

import numpy as np
import pandas as pd
import data_paths
from tqdm import tqdm
import tifffile
import pickle
import settings
import sys
import data_paths


def generate__train_image_list():
    x_img_path_list = []
    y = []

    df = pd.read_csv(data_paths.occurrences_train, sep=';', low_memory=False)

    species_map = {l: i for i, l in enumerate(df.species_glc_id.unique())}

    print("Reading CSV...")

    for index, row in tqdm(df.iterrows(), miniters=100):
        current_species_glc_id = row["species_glc_id"]
        current_patch_dirname = row["patch_dirname"]
        current_patch_id = row["patch_id"]

        target = np.zeros(len(species_map.keys()))
        target[species_map[current_species_glc_id]] = 1

        img_path =[current_patch_dirname, current_patch_id]
        
        x_img_path_list.append(img_path)
        y.append(target)
    
    print("Writing Data...")

    y = np.array(y)
    x_img_path_list = np.array(x_img_path_list, dtype=np.uint32)

    np.save(data_paths.image_train_set_x, x_img_path_list)
    np.save(data_paths.image_train_set_y, y)


#def generate_test_image_list():


if __name__ == '__main__':
    generate__train_image_list()
