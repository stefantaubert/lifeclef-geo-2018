'''
I have created a vector of the channels for each test/trainingsrow. 
Then I calculate the difference between the current testrow vector with every trainrow vector. 
Then I use the length of the resulting vectors to see which row of the trainset is the most similar to the testrow. 
The class with the highest probability is then the species of this trainrow. 
The other classes were obtained from other trainrows which were less similar to the testrow (descending with regard to their similarity).
Test-mrr: 0.0271210174024472
'''

import pandas as pd
import numpy as np
import multiprocessing as mp
from tqdm import tqdm

from geo.calculation.get_vector_length import get_vector_length
from geo.models.data_paths import vector_submission
from geo.data_paths import test
from geo.data_paths import train
from geo.data_paths import get_suffix_pro
from geo.preprocessing.preprocessing import create_datasets
from geo.postprocessing.submission_maker import make_submission_df
from geo.models.settings import TOP_N_SUBMISSION_RANKS
from geo.logging.log import log_start
from geo.logging.log import log_end

train_columns = [ 
        'chbio_1', 'chbio_2', 'chbio_3', 'chbio_4', 'chbio_5', 'chbio_6',
        'chbio_7', 'chbio_8', 'chbio_9', 'chbio_10', 'chbio_11', 'chbio_12',
        'chbio_13', 'chbio_14', 'chbio_15', 'chbio_16', 'chbio_17', 'chbio_18','chbio_19', 
        'etp', 'alti', 'awc_top', 'bs_top', 'cec_top', 'crusting', 'dgh', 'dimp', 'erodi', 'oc_top', 'pd_top', 'text',
        'proxi_eau_fast', 'clc', 'latitude', 'longitude'
    ]

def run_vector_model(use_multithread = True):
    log_start()
    print("Run model for testdata...")
    create_datasets()
    x_train = pd.read_csv(train)
    x_test = pd.read_csv(test)
    
    y = x_train["species_glc_id"]
    species = sorted(np.unique(y))
    species_count = len(species)
    print("Count of species", species_count)        

    test_glc_ids = x_test["patch_id"]
    x_train = x_train[train_columns]
    x_test = x_test[train_columns]

    x_train_matrix = x_train.as_matrix()
    to_predict_matrix = x_test.as_matrix()
    fake_propabilities = [(species_count - i) / species_count for i in range(species_count)]
    count_of_rows = len(to_predict_matrix)

    if use_multithread:
        num_cores = mp.cpu_count()
        print("Cpu count:", str(num_cores))
        predictions = []
        pool = mp.Pool(processes=num_cores)
        for row in range(count_of_rows):
            pool.apply_async(predict_row, args=(row,to_predict_matrix, x_train_matrix, fake_propabilities,y,), callback=predictions.append)
        pool.close()
        pool.join()
    else:
        predictions = []
        for row in tqdm(range(count_of_rows)):
            predictions.append(predict_row(row,to_predict_matrix, x_train_matrix, fake_propabilities,y))
    
    #sort after rows
    predictions = sorted(predictions)
    _, props = zip(*predictions)   
    result = np.array(props)
    assert len(predictions) == len(x_test.index)
    print("Finished.")

    print("Create test submission...")
    df = make_submission_df(TOP_N_SUBMISSION_RANKS, species, result, test_glc_ids)

    print("Save test submission...")
    df.to_csv(vector_submission, index=False, sep=";", header=None)
    print("Finished.", vector_submission)

    log_end("Vector Model","Suffix: {}\nTraincolumns: {}\n".format(get_suffix_pro(), ", ".join(train_columns)))

def predict_row(row_nr, to_predict_matrix, x_train_matrix, fake_propabilities, y):
    print("predict", str(row_nr))
    row = np.array(to_predict_matrix[row_nr])
    distances = []

    for j in range(len(x_train_matrix)):          
        train_row = x_train_matrix[j]
        distance = get_vector_length(train_row - row)
        distances.append(distance)
        
    _, species_sorted = zip(*sorted(zip(distances, list(y))))
    species_sorted = list(dict.fromkeys(species_sorted))
    fake_props = list(fake_propabilities)
    _, fake_propabilities_sorted = zip(*sorted(zip(species_sorted, fake_props)))

    print("Finished row", str(row_nr + 1))
    return (row_nr, fake_propabilities_sorted)

if __name__ == '__main__':
    run_vector_model()

    
# def predict_train(self):
#     ranks = []
#     x_train_matrix = x_train.as_matrix()
#     to_predict_matrix = x_train.as_matrix()
#     y_train_matrix = y.as_matrix()
    
#     for i in tqdm(range(len(to_predict_matrix))):
#         row = np.array(to_predict_matrix[i])
#         distances = []
#         for j in range(len(x_train_matrix)):
#             if i == j:
#                 continue
#             else:
#                 train_row = x_train_matrix[j]
#                 distance = get_vector_length(train_row - row)
#                 distances.append(distance)
        
#         _, species_sorted = zip(*sorted(zip(distances, list(y))))
#         #print(distances_sorted)
#         species_sorted = list(dict.fromkeys(species_sorted))
        
#         solution = y_train_matrix[i]
#         rank = species_sorted.index(solution) + 1 #da 1-basiert
#         ranks.append(rank)
#         print("Solutionindex:", rank)
#         print("Average-Rank:", np.mean(ranks))
#         print("Mrr:", mrr.mrr_score(ranks))
