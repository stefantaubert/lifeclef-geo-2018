from DataReader import read_data
from settings import *
import numpy as np
import data_paths
import pickle
from scipy.stats import rankdata
import pandas as pd
from sklearn.model_selection import train_test_split

def evaluate():
    x_text = np.load(data_paths.x_text)
    y = np.load(data_paths.y)
    species = np.load(data_paths.species_map)

    df = pd.read_csv(data_paths.occurrences_train, sep=';', low_memory=False)
    species = df.species_glc_id.unique()

    x_train, x_valid, y_train, y_valid = train_test_split(x_text, y, test_size=train_val_split, random_state=seed)

    y_predicted = np.load(data_paths.prediction)

    result = pd.DataFrame(columns=['glc_id', 'species_glc_id', 'probability', 'rank', 'real_species_glc_id'])

    for i in range(0, len(y_valid)):
        current_pred = y_predicted[i]
        current_glc_id = i
        # pred_ranked = pd.Series(current_pred).rank(ascending=False, method="min")
        pred_r = rankdata(current_pred, method="ordinal")
        # absteigend sortieren
        pred_r = len(species) - pred_r + 1
        glc_id_array = [int(current_glc_id)] * len(species)

        current_solution = y_valid[i]
        #ind = current_solution.index(1)
        i, = np.where(y_valid[i] == 1)
        assert len(i) == 1
        current_solution_species = int(species[i])
        sol_array = [current_solution_species] * len(species)

        percentile_list = pd.DataFrame({
            'glc_id': glc_id_array,
            'species_glc_id': species,
            'probability': current_pred,
            'rank': pred_r,
            'real_species_glc_id': sol_array,
        })

        # macht aus int float!
        # percentile_list = pd.DataFrame(np.column_stack([glc_id_array, train_species_ids, current_pred, pred_r]), columns=['glc_id', 'species_glc_id', 'probability', 'rank'])
        result = pd.concat([result, percentile_list], ignore_index=True)

    result = result.reindex(columns=('glc_id', 'species_glc_id', 'probability', 'rank', 'real_species_glc_id'))
    result = result[result.species_glc_id == result.real_species_glc_id]

    # print(result)

    print("Calculate MRR-Score...")
    sum = 0.0
    Q = len(result.index)

    #print("Dropped valpredictions because species was only in valset:", len(y_valid) - Q)

    # MRR berechnen
    for index, row in result.iterrows():
        sum += 1 / float(row["rank"])

    mrr_score = 1.0 / Q * sum
    print("MRR-Score:", mrr_score)

    result.drop(['real_species_glc_id'], axis=1, inplace=True)
    result.to_csv(data_paths.submission_val, index=False, sep=";")

if __name__ == '__main__':
    evaluate()
