import pandas as pd
import numpy as np
import time
from scipy.stats import rankdata
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import data_paths
import submission_maker
import evaluation
import DataReader
import settings
import time
import random
from collections import Counter
from tqdm import tqdm
from bisect import bisect_left

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
    counter = Counter(species_ids)
    y = np.load(data_paths.y_ids)
    print(list(counter.items()))
    countRows = len(x_text.index)

    species_map = []
    array = []
    for item, count in tqdm(sorted(counter.items())):
        array.append(count / countRows * 100 )
        species_map.append(int(item))

    np.save(data_paths.species_map_training, np.asarray(species_map))
    
    assert len(array) == len(set(species_ids))

    x_train, x_valid, y_train, y_valid = train_test_split(x_text, y, test_size=settings.train_val_split, random_state=settings.seed)
    print("Create prediction...")
    valid_row_count = len(x_valid)

    prediction = []
    for i in tqdm(range(valid_row_count)):
        prediction.append(array)

    print("Save prediction...")
    np.save(data_paths.prediction, prediction)


if __name__ == '__main__':
    start_time = time.time()

    # DataReader.read_and_write_data()
    run_Model()
    submission_maker.make_submission()
    evaluation.evaluate_with_mrr()

    print("Total duration:", time.time() - start_time, "s")