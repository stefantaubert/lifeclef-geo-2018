import module_support_main
import pandas as pd
import numpy as np
import time
import data_paths_main as data_paths
import submission_maker
import settings_main as settings
import time
import random
from tqdm import tqdm
import datetime
import main_preprocessing
import Log
from joblib import Parallel, delayed
import multiprocessing as mp

class Model():
    '''
    Assign complete random species for each testrow
    Test-mrr: 0.00155669467421085
    '''

    def __init__(self):
        random.seed = settings.seed
        main_preprocessing.create_datasets()
        x_text = pd.read_csv(data_paths.train)
        x_test = pd.read_csv(data_paths.test)
        
        y = x_text["species_glc_id"]
       
        # species_count = np.load(data_paths.y_array).shape[1]
        self.species_map = list(np.unique(y))
        #np.save(data_paths.xgb_species_map, classes_)
        # np.save(data_paths.xgb_species_map, self.species_map)
        # np.save(data_paths.xgb_glc_ids, self.x_valid["patch_id"])
        self.test_glc_ids = x_test["patch_id"]
        self.x_test = x_test

    def predict_test(self):
        print("Run model...")
        self.species_count = len(self.species_map)
        self.fake_propabilities = [(self.species_count - i) / self.species_count for i in range(self.species_count)]
        test_predictions = []
        for _ in tqdm(range(len(self.x_test.index))):
            test_predictions.append(self.get_random_prediction())

        print("Finished.")
        self.test_predictions = test_predictions

    def get_random_prediction(self):
        random_species = random.sample(self.species_map, self.species_count)
        probs = list(self.fake_propabilities)
        _, sorted_probs = zip(*sorted(zip(random_species, probs)))
        return sorted_probs

def run():
    start_time = time.time()
    start_datetime = datetime.datetime.now()
    print("Start:", start_datetime)
    m = Model()
    m.predict_test()
    print("Create test submission...")
    df = submission_maker.make_submission_df(settings.TOP_N_SUBMISSION_RANKS, m.species_map, m.test_predictions, m.test_glc_ids)
    print("Save test submission...")
    df.to_csv(data_paths.random_submission, index=False, sep=";", header=None)
    print("Finished.", data_paths.random_submission)
    end_date_time = datetime.datetime.now()
    print("End:", end_date_time)
    seconds = time.time() - start_time
    duration_min = round(seconds / 60, 2)
    print("Total duration:", duration_min, "min")
    log_text = str("{}\n--------------------\nStarted: {}\nFinished: {}\nDuration: {}min\nSuffix: {}\n".format
    (
        "Random Model",
        str(start_datetime), 
        str(end_date_time),
        str(duration_min),
        data_paths.get_suffix_pro(),
    ))
    log_text += "============================="
    Log.write(log_text)
    print(log_text)

if __name__ == '__main__':
    run()