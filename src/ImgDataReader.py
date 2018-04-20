import numpy as np
import pandas as pd
import data_paths
from tqdm import tqdm
import tifffile
import pickle
import global_settings as settings
import sys

#Try -> leeres numpy file erstellen, mit mmap mode öffnen, img appenden, speichern

count_channels = 33
image_dimension = 64

def read_img_data():
    
    df = pd.read_csv(data_paths.occurrences_train, sep=';', low_memory=False)
    
    species_map = {l: i for i, l in enumerate(df.species_glc_id.unique())}
    
    if settings.read_data_count > 0:
        df = df.head(settings.read_data_count)

    df = df.head(1)
    
    input_count = df.count

    x_img = np.memmap(data_paths.x_img_numpy, dtype='uint8', mode='r+', shape=(1, ))
    y = []

    for index, row in tqdm(df.iterrows(), miniters=100):
        current_species_glc_id = row["species_glc_id"]
        current_patch_dirname = row["patch_dirname"]
        current_patch_id = row["patch_id"]

        target = np.zeros(len(species_map.keys()))
        target[species_map[current_species_glc_id]] = 1

        img = tifffile.imread(data_paths.patch_train+'/{}/patch_{}.tif'.format(current_patch_dirname, current_patch_id))
        assert img.shape == (count_channels, image_dimension, image_dimension)
            
        print(index)
        x_img[index] = img

        y.append(target)

        
    x_img = np.array(x_img, dtype=np.uint8)
    print(x_img.shape)

    y = np.array(y)
    np.save(data_paths.y_img, y)
    pickle.dump(species_map, open(data_paths.species_map_img, 'wb'))



if __name__ == '__main__':
    read_img_data()

