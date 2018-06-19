'''
Assign complete random species for each testrow
Test-mrr: 0.00155669467421085
'''

import pandas as pd
import numpy as np
import time
import random
from tqdm import tqdm

from geo.models.settings import seed
from geo.models.settings import TOP_N_SUBMISSION_RANKS
from geo.models.data_paths import random_submission
from geo.preprocessing.preprocessing import create_datasets
from geo.postprocessing.submission_maker import make_submission_df
from geo.data_paths import train
from geo.data_paths import test
from geo.data_paths import get_suffix_pro
from geo.logging.log import log_start
from geo.logging.log import log_end

def run_random_model():
    log_start()
    random.seed = seed
    create_datasets()
    x_train = pd.read_csv(train)
    x_test = pd.read_csv(test)
    
    y = x_train["species_glc_id"]
    
    species_map = list(np.unique(y))
    test_glc_ids = x_test["patch_id"]
    x_test = x_test
    print("Run model...")
    species_count = len(species_map)
    fake_propabilities = [(species_count - i) / species_count for i in range(species_count)]
    test_predictions = []
    for _ in tqdm(range(len(x_test.index))):
        test_predictions.append(_get_random_prediction(species_map, species_count, fake_propabilities))
    print("Finished.")

    print("Create test submission...")
    df = make_submission_df(TOP_N_SUBMISSION_RANKS, species_map, test_predictions, test_glc_ids)

    print("Save test submission...")
    df.to_csv(random_submission, index=False, sep=";", header=None)
    print("Finished.", random_submission)
    
    log_end("Random Model", "Suffix: {}\n".format(get_suffix_pro()))

def _get_random_prediction(species_map, species_count, fake_propabilities):
    random_species = random.sample(species_map, species_count)
    probs = list(fake_propabilities)
    _, sorted_probs = zip(*sorted(zip(random_species, probs)))
    return sorted_probs

if __name__ == '__main__':
    run_random_model()