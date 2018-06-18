import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from collections import Counter

from geo.preprocessing.text_preprocessing import load_train
from geo.data_paths import most_common_values
from geo.settings import use_mean

cols_to_consider = ['chbio_1', 'chbio_2', 'chbio_3', 'chbio_4', 'chbio_5', 'chbio_6',
    'chbio_7', 'chbio_8', 'chbio_9', 'chbio_10', 'chbio_11', 'chbio_12',
    'chbio_13', 'chbio_14', 'chbio_15', 'chbio_16', 'chbio_17', 'chbio_18',
    'chbio_19', 'etp', 'alti',
    'awc_top', 'bs_top', 'cec_top', 'crusting', 'dgh', 'dimp', 'erodi', 'oc_top', 'pd_top', 'text',
    'proxi_eau_fast', 'clc',
    #'day', 'month', 'year',
    'latitude', 'longitude']

def load_most_common_values():
    assert os.path.exists(most_common_values)
    return pd.read_csv(most_common_values)

def extract_most_common_values():
    if not os.path.exists(most_common_values):
        _create()
    else: 
        print("Most common values already exist.")

def _get_most_common_value_matrix_df(use_mean):
    csv, species, _ = load_train()

    result_cols = cols_to_consider + ['occurence', 'species_glc_id']
    resulting_rows = []

    for specie in tqdm(species):
        specie_csv = csv[csv["species_glc_id"] == specie]
        row = []

        for col in cols_to_consider:
            if use_mean:
                val = np.mean(specie_csv[col])
            else:
                c = Counter(specie_csv[col])
                val, _ = c.most_common(1)[0]
            row.append(val)

        row.append(len(specie_csv.index))
        row.append(specie)
        resulting_rows.append(row)
        # print(np.amin(row))
        # print(np.amax(row))

    results_array = np.asarray(resulting_rows) #list to array to add to the dataframe as a new column
    result_ser = pd.DataFrame(results_array, columns=result_cols)   

    return result_ser

def _create():
    print("Getting most common values...")
    mean = True if use_mean==1 else False
    most_common_value_matrix = _get_most_common_value_matrix_df(mean)
    most_common_value_matrix.to_csv(most_common_values, index=False)

if __name__ == "__main__":
    _create()