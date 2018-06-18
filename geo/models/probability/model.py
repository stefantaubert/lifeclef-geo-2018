'''
Predict the same species for each row in testdata.
Those species are ordered descending after their occurences in trainset.
That means the most frequent species is always at rank one, the second at rank two and so on for all testrows.
Test-mrr: 0.0134454691619079
'''

import pandas as pd
import numpy as np
import time
import random
import datetime
from tqdm import tqdm

from geo.preprocessing.preprocessing import create_datasets
from geo.data_paths import test
from geo.preprocessing.species_occurences import extract_species_occurences
from geo.preprocessing.species_occurences import load_species_occurences
from geo.postprocessing.submission_maker import make_submission_df
from geo.models.settings import TOP_N_SUBMISSION_RANKS
from geo.models.data_paths import probability_submission
from geo.logging.log import log_start
from geo.logging.log import log_end

def run_probability_model():
    log_start()
    create_datasets()
    x_test = pd.read_csv(test)
    extract_species_occurences()
    species_occ = load_species_occurences()
    species = list(species_occ['species'])
    percents = list(species_occ['percents'])
    species_count = len(species)
    # create descending fake probabilities
    fake_probabilities = [(species_count - i) / species_count for i in range(species_count)]
    # sort after percents descending
    _, species_sorted = zip(*reversed(sorted(zip(percents, species))))
    # sort after species ascending
    species_map, probabilities_sorted = zip(*sorted(zip(species_sorted, list(fake_probabilities))))        
    test_glc_ids = x_test["patch_id"]
    x_test = x_test

    test_predictions = []
    for _ in tqdm(range(len(x_test.index))):
        test_predictions.append(probabilities_sorted)
    print("Finished.")

    print("Create test submission...")
    df = make_submission_df(TOP_N_SUBMISSION_RANKS, species_map, test_predictions, test_glc_ids)

    print("Save test submission...")
    df.to_csv(probability_submission, index=False, sep=";", header=None)
    print("Finished.", probability_submission)

    log_end("Probability Model")

if __name__ == '__main__':
    run_probability_model()