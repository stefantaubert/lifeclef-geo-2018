from DataReader import read_data
from settings import *
import numpy as np
import data_paths
import pickle
from tqdm import tqdm
from scipy.stats import rankdata
import pandas as pd
from sklearn.model_selection import train_test_split

def make_submission():
    print("Make submission...")    
    x_text = np.load(data_paths.x_text)
    y = np.load(data_paths.y)
    #species = np.load(data_paths.species_map)
    y_predicted = np.load(data_paths.prediction)
    df = pd.read_csv(data_paths.occurrences_train, sep=';', low_memory=False)
    species = df.species_glc_id.unique()
    result = pd.DataFrame(columns=['glc_id', 'species_glc_id', 'probability', 'rank', 'real_species_glc_id'])

    x_train, x_valid, y_train, y_valid = train_test_split(x_text, y, test_size=train_val_split, random_state=seed)

    sol_rows = []

    for i in tqdm(range(len(y_valid))):
        current_pred_array = y_predicted[i]
        current_glc_id = i

        pred_r = rankdata(current_pred_array, method="ordinal")
        # absteigend sortieren
        pred_r = len(species) - pred_r + 1
       
        i, = np.where(y_valid[i] == 1)
        assert len(i) == 1

        current_rank = int(pred_r[i])
        current_species = int(species[i])
        current_prediction = float(current_pred_array[i])

        sol_row = []
        sol_row.append(current_glc_id)
        sol_row.append(current_species)
        sol_row.append(current_prediction)
        sol_row.append(current_rank)

        sol_rows.append(sol_row)
        
    result_ser = pd.DataFrame(sol_rows, columns = ['glc_id', 'species_glc_id', 'probability', 'rank'])
    result_ser.to_csv(data_paths.submission_val, index=False)

if __name__ == '__main__':
    make_submission()