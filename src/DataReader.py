import numpy as np
import pandas as pd
import data_paths
from tqdm import tqdm
import tifffile
import pickle
import settings
import sys


def read_data():
    x_img = []
    x_text = []
    y = []

    df = pd.read_csv(data_paths.occurrences_train, sep=';', low_memory=False)

    species_map = {l: i for i, l in enumerate(df.species_glc_id.unique())}

    df = df.head(settings.read_data_count)

    for index, row in tqdm(df.iterrows(), miniters=100):
        current_species_glc_id = row["species_glc_id"]
        current_patch_dirname = row["patch_dirname"]
        current_patch_id = row["patch_id"]

        target = np.zeros(len(species_map.keys()))
        target[species_map[current_species_glc_id]] = 1

        img = tifffile.imread(data_paths.patch_train+'/{}/patch_{}.tif'.format(current_patch_dirname, current_patch_id))
        assert img.shape == (33, 64, 64)

        #Change datatype of img from uint8 to enable mean calculation
        img = np.array(img, dtype=np.uint16)

        csv_values = np.zeros(len(img))

        for i in range(len(img)):
            csv_values[i] = (img[i][31][31] + img[i][31][32] + img[i][32][31] + img[i][32][32])/4

        x_img.append(img)
        x_text.append(csv_values)
        y.append(target)
        #y.append(current_species_glc_id)

    #Change datatype back to uint8
    x_img = np.array(x_img, dtype=np.uint8)
    x_text = np.array(x_text)
    y = np.array(y)

    return species_map, x_img, x_text, y


if __name__ == '__main__':
    species_map, x_img, x_text, y = read_data()
    np.save(data_paths.x_img, x_img)
    np.save(data_paths.x_text, x_text)
    np.save(data_paths.y, y)
    pickle.dump(species_map, open(data_paths.species_map, 'wb'))
