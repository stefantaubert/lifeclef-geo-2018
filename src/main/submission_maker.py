from tqdm import tqdm
from scipy.stats import rankdata
import pandas as pd

def _make_submission_groups(top_n, groups_map, predictions, glc_ids, groups, props):
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

            species_s = [int(x) for x in species_s]
            classes.extend(species_s)

            cur_predictions.extend(group_species_count * [group_prediction])
        
        current_glc_id_array = len(classes) * [int(current_glc_id)]
        submissions = [list(a) for a in zip(current_glc_id_array, classes, cur_predictions, ranks)]
        submissions = submissions[:top_n]
        submission.extend(submissions)
    
    return submission

def _make_submission(top_n, classes, predictions, glc_ids):
    '''
    Erstellt eine Submission mit den Einträgen in folgendem Schema: glc_id, species_glc_id, probability, rank.
    Ausgabe zb: [[1, "9", 0.5, 1], [1, "3", 0.6, 2], [2, "9", 0.7, 1], [2, "3", 0.6, 2]]
    '''
    assert len(glc_ids) == len(predictions)

    count_predictions = len(predictions)
    count_species = len(classes)
    assert top_n <= count_species
    submission = []

    for i in tqdm(range(count_predictions)):
        species = list(classes)
        current_glc_id = glc_ids[i]
        current_predictions = predictions[i]
        assert len(current_predictions) == count_species

        current_ranks = rankdata(current_predictions, method="ordinal")
        # rang 100,99,98 zu rang 1,2,3 machen
        current_ranks = count_species - current_ranks + 1
        current_glc_id_array = count_species * [current_glc_id]

        # sort after rank
        current_ranks, species, current_predictions = zip(*sorted(zip(current_ranks, species, current_predictions)))

        submissions = [list(a) for a in zip(current_glc_id_array, species, current_predictions, current_ranks)]

        submissions = submissions[:top_n]
        submission.extend(submissions)
    
    return submission

def _get_df(submission):
    submission_df = pd.DataFrame(submission, columns = ['patch_id', 'species_glc_id', 'probability', 'rank'])
    return submission_df

def make_submission_df(top_n, classes, predictions, glc_ids):
    submission = _make_submission(top_n, classes, predictions, glc_ids)
    return _get_df(submission)

def make_submission_groups_df(top_n, groups, predictions, glc_ids, groups_dict, props):
    submission = _make_submission_groups(top_n, groups, predictions, glc_ids, groups_dict, props)
    return _get_df(submission)
