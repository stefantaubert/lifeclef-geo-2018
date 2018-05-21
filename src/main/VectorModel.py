import module_support_main
import pandas as pd
import numpy as np
import data_paths_main as data_paths
import settings_main as settings
import submission_maker
import submission
from tqdm import tqdm
import get_ranks
import mrr
import time
import main_preprocessing
import Log
import datetime
import math
import mrr
import multiprocessing as mp

class Model():
    def __init__(self):
        main_preprocessing.create_datasets()
        x_train = pd.read_csv(data_paths.train)
        x_test = pd.read_csv(data_paths.test)#, nrows = 7)
      
        self.y = x_train["species_glc_id"]
        self.species = sorted(np.unique(self.y))
        self.species_count = len(self.species)
        print("Count of species", self.species_count)        

        self.train_columns = [ 
            'chbio_1', 'chbio_2', 'chbio_3', 'chbio_4', 'chbio_5', 'chbio_6',
            'chbio_7', 'chbio_8', 'chbio_9', 'chbio_10', 'chbio_11', 'chbio_12',
            'chbio_13', 'chbio_14', 'chbio_15', 'chbio_16', 'chbio_17', 'chbio_18','chbio_19', 
            'etp', 'alti', 'awc_top', 'bs_top', 'cec_top', 'crusting', 'dgh', 'dimp', 'erodi', 'oc_top', 'pd_top', 'text',
            'proxi_eau_fast', 'clc', 'latitude', 'longitude'
        ]
        
        self.test_glc_ids = x_test["patch_id"]
        self.x_train = x_train[self.train_columns]
        self.x_test = x_test[self.train_columns]

    def get_vector_length(self, v):
        summ = 0

        for num in v:
            summ += num * num

        distance = math.sqrt(summ)
        
        return distance

    def predict_train(self):
        ranks = []
        x_train_matrix = self.x_train.as_matrix()
        to_predict_matrix = self.x_train.as_matrix()
        y_train_matrix = self.y.as_matrix()
        
        for i in tqdm(range(len(to_predict_matrix))):
            row = np.array(to_predict_matrix[i])
            distances = []
            for j in range(len(x_train_matrix)):
                if i == j:
                    continue
                else:
                    train_row = x_train_matrix[j]
                    distance = self.get_vector_length(train_row - row)
                    distances.append(distance)
            
            _, species_sorted = zip(*sorted(zip(distances, list(self.y))))
            #print(distances_sorted)
            species_sorted = list(dict.fromkeys(species_sorted))
            
            solution = y_train_matrix[i]
            rank = species_sorted.index(solution) + 1 #da 1-basiert
            ranks.append(rank)
            print("Solutionindex:", rank)
            print("Average-Rank:", np.mean(ranks))
            print("Mrr:", mrr.mrr_score(ranks))

    def predict_test(self, use_multithread = True):
        self.x_train_matrix = self.x_train.as_matrix()
        self.to_predict_matrix = self.x_test.as_matrix()
        self.fake_propabilities = [(self.species_count - i) / self.species_count for i in range(self.species_count)]
        count_of_rows = len(self.to_predict_matrix)

        if use_multithread:
            num_cores = mp.cpu_count()
            print("Cpu count:", str(num_cores))
            predictions = []
            pool = mp.Pool(processes=num_cores)
            for row in range(count_of_rows):
                pool.apply_async(self.predict_row, args=(row,), callback=predictions.append)
            pool.close()
            pool.join()
            #print(predictions)
        else:
            predictions = []
            for row in tqdm(range(count_of_rows)):
                predictions.append(self.predict_row(row))
        
        #sort after rows
        #print(predictions)
        predictions = sorted(predictions)
        rows, props = zip(*predictions)
        #print(rows)
        #print(props)
        #print(np.array(props))
        #print("Saving test predictions...")
        result = np.array(props)
        assert len(predictions) == len(self.x_test.index)
        return result

    def predict_row(self, row_nr):
        row = np.array(self.to_predict_matrix[row_nr])
        distances = []

        for j in range(len(self.x_train_matrix)):          
            train_row = self.x_train_matrix[j]
            distance = self.get_vector_length(train_row - row)
            distances.append(distance)
            
        distances_sorted, species_sorted = zip(*sorted(zip(distances, list(self.y))))
        #print(distances_sorted[:7])
        species_sorted = list(dict.fromkeys(species_sorted))
        fake_props = list(self.fake_propabilities)
        # print("NEW ITERATION------------------------------------")
        #print(species_sorted[:100])
        # print(fake_props[:100])

        _, fake_propabilities_sorted = zip(*sorted(zip(species_sorted, fake_props)))

        #print(species_map[:100])
        # print(fake_propabilities_sorted[:100])

        print("Finished row", str(row_nr + 1))
        return (row_nr, fake_propabilities_sorted)
        #print(probabilities)

    def make_test_submission(self):
        print("Run model for testdata...")
        predictions = self.predict_test(use_multithread=True)
        print("Finished.")
        print("Create test submission...")
        df = submission_maker.make_submission_df(settings.TOP_N_SUBMISSION_RANKS, self.species, predictions, self.test_glc_ids)
        print("Save test submission...")
        df.to_csv(data_paths.vector_test_submission, index=False, sep=";", header=None)
        print("Finished.", data_paths.vector_test_submission)

def run():
    start_time = time.time()
    start_datetime = datetime.datetime.now()
    print("Start:", start_datetime)
    m = Model()
    m.make_test_submission()
    end_date_time = datetime.datetime.now()
    print("End:", end_date_time)
    seconds = time.time() - start_time
    duration_min = round(seconds / 60, 2)
    print("Total duration:", duration_min, "min")
    log_text = str("{}\n--------------------\nStarted: {}\nFinished: {}\nDuration: {} min\nSuffix: {}\n".format
    (
        "Vector Model",
        str(start_datetime), 
        str(end_date_time),
        str(duration_min),
        data_paths.get_suffix_pro(),
    ))
    log_text += "============================="
    Log.write(log_text)
    print("#### LOG ####")
    print(log_text)

if __name__ == '__main__':
    run()