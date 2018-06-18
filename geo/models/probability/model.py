import module_support_main
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
import data_paths_main as data_paths
import submission_maker
import settings_main as settings
import time
import random
from tqdm import tqdm
import main_preprocessing
import SpeciesOccurences
import data_paths_analysis
import datetime
import Log

class Model():
    '''
    Predict the same species for each row in testdata.
    Those species are ordered descending after their occurences in trainset.
    That means the most frequent species is always at rank one, the second at rank two and so on for all testrows.
    Test-mrr: 0.0134454691619079
    '''
    def __init__(self):
        main_preprocessing.create_datasets()
        x_test = pd.read_csv(data_paths.test)
        SpeciesOccurences.create()
        species_occ = pd.read_csv(data_paths_analysis.species_occurences)
        species = list(species_occ['species'])
        percents = list(species_occ['percents'])
        self.species_count = len(species)
        # create descending fake probabilities
        self.fake_probabilities = [(self.species_count - i) / self.species_count for i in range(self.species_count)]
        # sort after percents descending
        _, species_sorted = zip(*reversed(sorted(zip(percents, species))))
        # sort after species ascending
        self.species_map, self.probabilities_sorted = zip(*sorted(zip(species_sorted, list(self.fake_probabilities))))        
        self.test_glc_ids = x_test["patch_id"]
        self.x_test = x_test

    def predict_test(self):
        self.test_predictions = []
        for _ in tqdm(range(len(self.x_test.index))):
            self.test_predictions.append(self.probabilities_sorted)
        print("Finished.")
    

def run():
    start_time = time.time()
    start_datetime = datetime.datetime.now()
    print("Start:", start_datetime)
    m = Model()
    m.predict_test()
    print("Create test submission...")
    df = submission_maker.make_submission_df(settings.TOP_N_SUBMISSION_RANKS, m.species_map, m.test_predictions, m.test_glc_ids)
    print("Save test submission...")
    df.to_csv(data_paths.probability_submission, index=False, sep=";", header=None)
    print("Finished.", data_paths.probability_submission)
    end_date_time = datetime.datetime.now()
    print("End:", end_date_time)
    seconds = time.time() - start_time
    duration_min = round(seconds / 60, 2)
    print("Total duration:", duration_min, "min")
    log_text = str("{}\n--------------------\nStarted: {}\nFinished: {}\nDuration: {}min\n".format
    (
        "Probability Model",
        str(start_datetime),
        str(end_date_time),
        str(duration_min),
    ))
    log_text += "============================="
    Log.write(log_text)
    print(log_text)


if __name__ == '__main__':
    run()