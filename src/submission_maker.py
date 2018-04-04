from DataReader import read_data
from settings import *
import numpy as np
import data_paths
import pickle
from scipy.stats import rankdata
import pandas as pd
from sklearn.model_selection import train_test_split

def make_submission():
    x_text = np.load(data_paths.x_text)
    y = np.load(data_paths.y)
    #species = np.load(data_paths.species_map)
    y_predicted = np.load(data_paths.prediction)
    df = pd.read_csv(data_paths.occurrences_train, sep=';', low_memory=False)
    species = df.species_glc_id.unique()
    result = pd.DataFrame(columns=['glc_id', 'species_glc_id', 'probability', 'rank', 'real_species_glc_id'])

    x_train, x_valid, y_train, y_valid = train_test_split(x_text, y, test_size=train_val_split, random_state=seed)

    sol_rows = []
    for i in range(0, len(y_valid)):
        current_pred = y_predicted[i]

        current_glc_id = i

        pred_r = rankdata(current_pred, method="ordinal")
        # absteigend sortieren
        pred_r = len(species) - pred_r + 1
       
        i, = np.where(y_valid[i] == 1)
        assert len(i) == 1

        current_rank = pred_r[i]
        current_species = int(species[i])
        current_prediction = current_pred[i]

        sol_row = []
        sol_row.append(current_glc_id)
        sol_row.append(current_species)
        sol_row.append(current_prediction)
        sol_row.append(current_rank)

        # macht aus int float!
        # percentile_list = pd.DataFrame(np.column_stack([glc_id_array, train_species_ids, current_pred, pred_r]), columns=['glc_id', 'species_glc_id', 'probability', 'rank'])
        sol_rows.append(sol_row)
        
    results_array=np.asarray(sol_rows) #list to array to add to the dataframe as a new column

    result_ser =pd.Series(results_array, columns=['glc_id', 'species_glc_id', 'probability', 'rank', 'real_species_glc_id'])
    print(result_ser)

make_submission()