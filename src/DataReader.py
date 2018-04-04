import numpy as np
import pandas as pd
import data_paths
from tqdm import tqdm
import tifffile
import pickle
import settings
import sys

count_channels = 33
image_dimension = 64

def read_data():
    x_img = []
    x_text = []
    y = []

    df = pd.read_csv(data_paths.occurrences_train, sep=';', low_memory=False)
    result_cols = [  'chbio_1', 'chbio_2', 'chbio_3', 'chbio_4', 'chbio_5', 'chbio_6',
    'chbio_7', 'chbio_8', 'chbio_9', 'chbio_10', 'chbio_11', 'chbio_12',
    'chbio_13', 'chbio_14', 'chbio_15', 'chbio_16', 'chbio_17', 'chbio_18',
    'chbio_19', 'etp', 'alti',
    'awc_top', 'bs_top', 'cec_top', 'crusting', 'dgh', 'dimp', 'erodi', 'oc_top', 'pd_top', 'text',
    'proxi_eau_fast', 'clc', 'day', 'month', 'year', 'latitude', 'longitude', 'patch_id', 'species_glc_id',
                                     ]
    result = pd.DataFrame(columns=result_cols)

    result_col_count = len(result_cols)

    species_map = {l: i for i, l in enumerate(df.species_glc_id.unique())}
    
    if settings.read_data_count > 0:
        df = df.head(settings.read_data_count)

    for index, row in tqdm(df.iterrows(), miniters=100):
        current_species_glc_id = row["species_glc_id"]
        current_patch_dirname = row["patch_dirname"]
        current_patch_id = row["patch_id"]

        target = np.zeros(len(species_map.keys()))
        target[species_map[current_species_glc_id]] = 1

        img = tifffile.imread(data_paths.patch_train+'/{}/patch_{}.tif'.format(current_patch_dirname, current_patch_id))
        assert img.shape == (count_channels, image_dimension, image_dimension)

        #Change datatype of img from uint8 to enable mean calculation
        img = np.array(img, dtype=np.uint16)

        csv_values = []

        for i in range(len(img)):
            csv_values.append((img[i][31][31] + img[i][31][32] + img[i][32][31] + img[i][32][32])/4)

        csv_values.append(row.day)
        csv_values.append(row.month)
        csv_values.append(row.year)
        csv_values.append(row.Latitude)
        csv_values.append(row.Longitude)
        # patch_id != index, da zwischendrin immer mal zeilen mit höheren patch_ids vorkommen
        csv_values.append(row.patch_id)
        csv_values.append(row.species_glc_id)

        #x_img.append(img)
        #x_text.append(csv_values)
        result.loc[index] = csv_values
        #y.append(target)
        #y.append(current_species_glc_id)

    result.to_csv(data_paths.occurrences_train_gen, index=False)
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
