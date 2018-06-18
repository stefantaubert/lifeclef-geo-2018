import numpy as np
import pandas as pd
from tqdm import tqdm
import tifffile
import pickle
import os

from geo.preprocessing.pixel_value_extraction import get_pixel_value
from geo.data_paths import occurrences_train
from geo.data_paths import occurrences_test
from geo.data_paths import occurrences_test_gen
from geo.data_paths import occurrences_train_gen
from geo.data_paths import patch_train
from geo.data_paths import patch_test
from geo.settings import pixel_count

count_channels = 33
image_dimension = 64
result_cols = [
                'chbio_1', 'chbio_2', 'chbio_3', 'chbio_4', 'chbio_5', 'chbio_6',
                'chbio_7', 'chbio_8', 'chbio_9', 'chbio_10', 'chbio_11', 'chbio_12',
                'chbio_13', 'chbio_14', 'chbio_15', 'chbio_16', 'chbio_17', 'chbio_18','chbio_19',
                'etp', 'alti','awc_top', 'bs_top', 'cec_top', 'crusting', 'dgh', 'dimp', 'erodi', 'oc_top', 'pd_top', 'text',
                'proxi_eau_fast', 'clc', 
                'day', 'month', 'year', 'latitude', 'longitude', 'patch_id'
            ]

def load_occurences_train():
    assert os.path.exists(occurrences_train_gen)
    csv = pd.read_csv(occurrences_train_gen)
    return csv

def load_occurences_test():
    assert os.path.exists(occurrences_test_gen)
    csv = pd.read_csv(occurrences_test_gen)
    return csv

def extract_occurences_train():
    if not os.path.exists(occurrences_train_gen):
        _create_train()
    else:
        print("Occurences train already exists.")

def extract_occurences_test():
    if not os.path.exists(occurrences_test_gen):
        _create_test()
    else:
        print("Occurences test already exists.")

def get_dataset_df(path_occurences, batch_dir):
    x_text = []
    df = pd.read_csv(path_occurences, sep=';', low_memory=False)
    species_id_column_name = 'species_glc_id'
    is_train = species_id_column_name in df.columns.values
    columns = list(result_cols)

    if is_train:
        columns.append(species_id_column_name)
    
    for _, row in tqdm(df.iterrows(), miniters=100):
        current_patch_dirname = row["patch_dirname"]
        current_patch_id = row["patch_id"]

        img = tifffile.imread(batch_dir+'/{}/patch_{}.tif'.format(current_patch_dirname, current_patch_id))
        assert img.shape == (count_channels, image_dimension, image_dimension)

        csv_values = []

        for i in range(len(img)):
            mean = get_pixel_value(img[i], pixel_count)
            csv_values.append(mean)

        csv_values.append(row.day)
        csv_values.append(row.month)
        csv_values.append(row.year)
        csv_values.append(row.Latitude)
        csv_values.append(row.Longitude)

        # patch_id != index, da zwischendrin immer mal zeilen mit höheren patch_ids vorkommen
        csv_values.append(row.patch_id)
        
        if is_train:
            csv_values.append(row.species_glc_id)

        x_text.append(csv_values)

    results_array = np.asarray(x_text) #list to array to add to the dataframe as a new column
    result_ser = pd.DataFrame(results_array, columns=columns)
    
    return result_ser

    
def _create_train():
    df = get_dataset_df(occurrences_train, patch_train)
    df.to_csv(occurrences_train_gen, index=False)

def _create_test():
    df = get_dataset_df(occurrences_test, patch_test)
    df.to_csv(occurrences_test_gen, index=False)
