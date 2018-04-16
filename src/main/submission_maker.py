import numpy as np
import data_paths
import pickle
from tqdm import tqdm
from scipy.stats import rankdata
import pandas as pd
from sklearn.model_selection import train_test_split
import json 

def make_submission_array(classes, predictions, glc_ids):
    '''
    Erstellt eine Submission mit den Einträgen in folgendem Schema: glc_id, species_glc_id, probability, rank.
    Ausgabe zb: [[1, "9", 0.5, 1], [1, "3", 0.6, 2], [2, "9", 0.7, 1], [2, "3", 0.6, 2]]
    Die glc_id ist dabei 1-basiert.
    '''
    assert len(glc_ids) == len(predictions)

    count_predictions = len(predictions)
    count_classes = len(classes)
    submission = []

    for i in tqdm(range(count_predictions)):
        current_glc_id = glc_ids[i]
        current_predictions = predictions[i]
        assert len(current_predictions) == count_classes

        current_ranks = rankdata(current_predictions, method="ordinal")
        # rang 100,99,98 zu rang 1,2,3 machen
        current_ranks = count_classes - current_ranks + 1
        current_glc_id_array = count_classes * [current_glc_id]
        submissions = [list(a) for a in zip(current_glc_id_array, classes, current_predictions, current_ranks)]
        submission.extend(submissions)
    
    return submission

def make_submission_df(classes, predictions, glc_ids):
    submission = make_submission_array(classes, predictions, glc_ids)
    submission_df = pd.DataFrame(submission, columns = ['glc_id', 'species_glc_id', 'probability', 'rank'])
    return submission_df

def make_submission():
    print("Make submission...")
    # y_ids = np.load(data_paths.y_ids)

    # np.save(data_paths.species_map_training, np.unique(y_ids))

    classes = np.load(data_paths.species_map_training)

    predictions = np.load(data_paths.prediction)
    df = make_submission_df(classes, predictions, glc_ids)

    print("Save submission...")
    df.to_csv(data_paths.submission_val, index=False)

    # y = None
    # with open(data_paths.species_map, 'rb') as f:
    #     y = pickle.load(f)
        

def make_submission_for_current_training():
    print("Make submission...")

    classes = np.load(data_paths.current_training_species_map)

    predictions = np.load(data_paths.current_training_results)
    glc_ids = np.load(data_paths.current_training_glc_ids)
    df = make_submission_df(classes, predictions, glc_ids)

    print("Save submission...")
    df.to_csv(data_paths.current_training_submission, index=False)

if __name__ == '__main__':
    make_submission_for_current_training()