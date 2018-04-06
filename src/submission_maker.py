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
    #x_text = np.load(data_paths.x_text)
    
    x_text = pd.read_csv(data_paths.occurrences_train_gen)
    species = x_text.species_glc_id.unique()
    #y = x_text.species_glc_id
    x_text = x_text[['chbio_1', 'chbio_5', 'chbio_6','month', 'latitude', 'longitude']]

    y = np.load(data_paths.y_array)
    #species = np.load(data_paths.species_map)
    y_predicted = np.load(data_paths.prediction)
    x_train, x_valid, y_train, y_valid = train_test_split(x_text, y, test_size=train_val_split, random_state=seed)

    # print("Validationset rows after removing unique species:", len(x_valid.index))

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

    # <glc_id;species_glc_id;probability;rank>        
    result_ser = pd.DataFrame(sol_rows, columns = ['glc_id', 'species_glc_id', 'probability', 'rank'])
    result_ser.to_csv(data_paths.submission_val, index=False)

if __name__ == '__main__':
    make_submission()