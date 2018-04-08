import pandas as pd
import numpy as np
import time
from scipy.stats import rankdata
from sklearn.model_selection import train_test_split
#from xgboost import XGBClassifier
import data_paths
import pickle
import submission_maker
import evaluation
import DataReader
import settings
import time
import random
from collections import Counter
from tqdm import tqdm
from bisect import bisect_left
np.set_printoptions(threshold=np.nan)
random.seed = settings.seed

count_most_common = 531


def binary_search(a, x, lo=0, hi=None):  # can't use a to specify default for hi
    hi = hi if hi is not None else len(a)  # hi defaults to len(a)   
    pos = bisect_left(a, x, lo, hi)  # find insertion position
    return (pos if pos != hi and a[pos] == x else -1)  # don't walk off the end

def run_Model():
    print("Run model...")
    #x_text = np.load(data_paths.x_text)

    x_text = pd.read_csv(data_paths.occurrences_train_gen)
    species_ids = x_text["species_glc_id"].values
    y_array = np.load(data_paths.y_array)

    # with open(data_paths.species_map, 'rb') as f:
    #     y = pickle.load(f)

    # print(y)
    # y2 = np.unique(y_ids)
    # print(y2)

    # y1 = list(y.keys())

    # # for i in range(len(y2)):
    # #     print(y1[i])
    # #     print(y2[i])
    # #     assert y1[i] == y2[i]

    # for k_from_dict, v in y.items():
    #     k_from_keylist = y1[v]
    #     if k_from_keylist != k_from_dict:
    #         print("Fail------------")
    
    with open(data_paths.species_map, 'rb') as f:
        species_map = pickle.load(f)

    classes_ = list(species_map.keys())
    np.save(data_paths.species_map_training, classes_)

    for index, row in x_text.iterrows():
        current_species = row["species_glc_id"]
        y = y_array[index]
        assert len(y) == len(classes_)

        for i in range(len(y_array)):
            current_sol = y[i]
            if current_sol == 1:
                assert classes_[i] == current_species

    x_train, x_valid, y_train, y_valid = train_test_split(x_text, y_array, test_size=settings.train_val_split, random_state=settings.seed)
    print(y_valid)
    print("Save prediction...")
    np.save(data_paths.prediction, y_valid)


if __name__ == '__main__':
    start_time = time.time()

    #DataReader.read_and_write_data()
    run_Model()
    submission_maker.make_submission()
    evaluation.evaluate_with_mrr()

    print("Total duration:", time.time() - start_time, "s")