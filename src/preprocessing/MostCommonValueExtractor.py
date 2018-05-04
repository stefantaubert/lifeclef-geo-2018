import module_support_pre
import pandas as pd
import numpy as np
import data_paths_pre as data_paths
import TextPreprocessing
from tqdm import tqdm
from collections import Counter
import os

def load():
    assert os.path.exists(data_paths.most_common_values)
    
    return pd.read_csv(data_paths.most_common_values)

def extract():
    if not os.path.exists(data_paths.most_common_values):
        MostCommonValueExtractor()._create()
    else: 
        print("Most common values already exist.")

class MostCommonValueExtractor():
    def __init__(self):
        csv, species, species_c = TextPreprocessing.load_train()
        self.csv = csv
        self.species = species
        self.species_count = species_c

        self.cols_to_consider = ['chbio_1', 'chbio_2', 'chbio_3', 'chbio_4', 'chbio_5', 'chbio_6',
            'chbio_7', 'chbio_8', 'chbio_9', 'chbio_10', 'chbio_11', 'chbio_12',
            'chbio_13', 'chbio_14', 'chbio_15', 'chbio_16', 'chbio_17', 'chbio_18',
            'chbio_19', 'etp', 'alti',
            'awc_top', 'bs_top', 'cec_top', 'crusting', 'dgh', 'dimp', 'erodi', 'oc_top', 'pd_top', 'text',
            'proxi_eau_fast', 'clc',
            #'day', 'month', 'year',
            'latitude', 'longitude']

    def _get_most_common_value_matrix_df(self, use_mean):
        result_cols = self.cols_to_consider + ['occurence', 'species_glc_id']
        resulting_rows = []

        for specie in tqdm(self.species):
            specie_csv = self.csv[self.csv["species_glc_id"] == specie]
            row = []

            for col in self.cols_to_consider:
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

    def _create(self):
        print("Getting most common values...")
        most_common_value_matrix = self._get_most_common_value_matrix_df(use_mean=True)
        most_common_value_matrix.to_csv(data_paths.most_common_values, index=False)

if __name__ == "__main__":
    MostCommonValueExtractor()._create()