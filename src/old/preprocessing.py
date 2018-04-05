import time
import pandas as pd
import numpy as np
import time
import pandas as pd

import data_paths

# Extrahiert alle wichtigen Spalten die für das Training notwending sind (glc_id, Features und species_glc_id) und entfernt ungültige Zeilen

important_columns = [
    'glc_id','species_glc_id',
    'chbio_1', 'chbio_2', 'chbio_3', 'chbio_4', 'chbio_5', 'chbio_6',
    'chbio_7', 'chbio_8', 'chbio_9', 'chbio_10', 'chbio_11', 'chbio_12',
    'chbio_13', 'chbio_14', 'chbio_15', 'chbio_16', 'chbio_17', 'chbio_18',
    'chbio_19', 'alti',
    'awc_top', 'bs_top', 'cec_top', 'crusting', 'dgh', 'dimp', 'erodi', 'oc_top', 'pd_top', 'text',
    'proxi_eau_fast', 'clc'
]

x_train = pd.read_csv(data_paths.occurrences_train, sep=';', low_memory=False)

x_train["glc_id"] = x_train["patch_id"]

x_train = x_train[important_columns]
old_rowcount = len(x_train.index)
old_species_count =  len(set(x_train["species_glc_id"].values))

# alle Zeilen entfernen, die in einer Spalte den Wert 'nan' besitzen
x_train = x_train.dropna(axis=0, how='any')

species_count = len(set(x_train["species_glc_id"].values))

print("Count of dropped rows with 'nan':", old_rowcount - len(x_train.index), "of", old_rowcount)

# x_train.to_csv(data_paths.train_features, index=False)
# x_train = pd.read_csv(data_paths.features_train)
# print("All species:", species)
print("Different species count:", species_count)
print("Count of dropped species:", old_species_count - species_count)

x_train.to_csv(data_paths.features_train, index=False)
print(x_train.head())
