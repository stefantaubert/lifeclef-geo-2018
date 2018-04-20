import numpy as np
import data_paths
import pickle
from tqdm import tqdm
from scipy.stats import rankdata
import pandas as pd
from sklearn.model_selection import train_test_split
import json 

def make_submission_groups(groups_map, predictions, glc_ids, groups, props):
    '''
    Erstellt eine Submission mit den Einträgen in folgendem Schema: glc_id, species_glc_id, probability, rank.
    Ausgabe zb: [[1, "9", 0.5, 1], [1, "3", 0.6, 2], [2, "9", 0.7, 1], [2, "3", 0.6, 2]]
    Dabei werden die Gruppen in die einzelnen Species aufgelöst und die Species mit der höchsten Auftrittswahrscheinlichkeit erhält den kleinsten Rang.
    '''
    assert len(glc_ids) == len(predictions)

    count_predictions = len(predictions)
    count_groups = len(groups_map)
    submission = []

    for i in tqdm(range(count_predictions)):
        current_glc_id = glc_ids[i]
        current_predictions = predictions[i]
        assert len(current_predictions) == count_groups

        group_predictions_s, groups_map_s = zip(*reversed(sorted(zip(current_predictions, groups_map))))
       
        ranks = []
        cur_predictions = []
        rank_counter = 0
        classes = []
        ### For each species in current group: create ranks
        for j in range(len(groups_map_s)):
            group_species = groups[groups_map_s[j]]
            group_species_count = len(group_species)
            species_props = []
            group_prediction = group_predictions_s[j]

            for species in group_species:
                species_props.append(props[species])

            _, species_s = zip(*reversed(sorted(zip(species_props, group_species))))

            for species in species_s:
                rank_counter += 1
                ranks.append(rank_counter)

            classes.extend(species_s)

            cur_predictions.extend(group_species_count * [group_prediction])
        
        current_glc_id_array = len(classes) * [current_glc_id]
        submissions = [list(a) for a in zip(current_glc_id_array, classes, cur_predictions, ranks)]
        submission.extend(submissions)
    
    return submission

def make_submission_array(classes, predictions, glc_ids):
    '''
    Erstellt eine Submission mit den Einträgen in folgendem Schema: glc_id, species_glc_id, probability, rank.
    Ausgabe zb: [[1, "9", 0.5, 1], [1, "3", 0.6, 2], [2, "9", 0.7, 1], [2, "3", 0.6, 2]]
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

def make_xgb_submission():
    print("Make submission...")

    classes = np.load(data_paths.xgb_species_map)

    predictions = np.load(data_paths.xgb_prediction)
    glc_ids = np.load(data_paths.xgb_glc_ids)
    df = make_submission_df(classes, predictions, glc_ids)

    print("Save submission...")
    df.to_csv(data_paths.xgb_submission, index=False)

def make_xgb_groups_submission():
    print("Make submission...")

    groups = np.load(data_paths.xgb_species_map)

    predictions = np.load(data_paths.xgb_prediction)
    glc_ids = np.load(data_paths.xgb_glc_ids)
    species_occ = pd.read_csv(data_paths.xgb_species_occurences)
    named_groups = pd.read_csv(data_paths.xgb_named_groups)
    
    species_occ_dict = {}
    for _, row in species_occ.iterrows():
        species_occ_dict[str(row["species"])] = float(row["percents"])

    data = make_submission_groups(groups, predictions, glc_ids, named_groups, species_occ_dict)
    submission_df = pd.DataFrame(data, columns = ['glc_id', 'species_glc_id', 'probability', 'rank'])
    
    print("Save submission...")
    submission_df.to_csv(data_paths.current_training_submission, index=False)

def make_submission_for_current_training():
    print("Make submission...")

    classes = np.load(data_paths.current_training_species_map)

    predictions = np.load(data_paths.current_training_results)
    glc_ids = np.load(data_paths.current_training_glc_ids)
    submission = make_submission_array(classes, predictions, glc_ids,)
    submission_df = pd.DataFrame(submission, columns = ['glc_id', 'species_glc_id', 'probability', 'rank'])
    
    print("Save submission...")
    submission_df.to_csv(data_paths.current_training_submission, index=False)

if __name__ == '__main__':
    make_submission_for_current_training()