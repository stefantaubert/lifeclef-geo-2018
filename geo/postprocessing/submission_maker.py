import pandas as pd
from tqdm import tqdm
from scipy.stats import rankdata

def _make_submission_groups(top_n, groups_map, predictions, glc_ids, groups, props):
    '''
    Creates a submission with the entries in the following schema: glc_id, species_glc_id, probability, rank.
    For example: [[1, 9, 0.5, 1], [1, 3, 0.6, 2], [2, 9, 0.7, 1], [2, 3, 0.6, 2]]
    In this case the groups will be resolved to the single species whereas the most common species get the lowest rank.
    
    Keyword arguments:
    top_n -- considers only the top_n-predictions
    groups_map -- contains a simple list of all possible groups
    predictions -- contains the predicted probabilities for each glc_id
    glc_ids -- contains all glc_ids of the dataset which was predicted
    groups -- contains a dictionary in form of: {group1: [species of group1], group2: [species of group2], ...} where the keys represent the groups_map. (created with main_preprocessing.extract_groups())
    props -- contains a dictionary which holds the probabilities for each species (created with SpeciesOccurences.create())
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
    Creates a submission with the entries in the following schema: glc_id, species_glc_id, probability, rank.
    For example: [[1, 9, 0.5, 1], [1, 3, 0.6, 2], [2, 9, 0.7, 1], [2, 3, 0.6, 2]]

    Keyword arguments:
    top_n -- considers only the top_n-predictions
    classes -- contains a list of all possible classes
    predictions -- contains the predicted probabilities for each glc_id
    glc_ids -- contains all glc_ids of the dataset which was predicted
    '''

    assert len(glc_ids) == len(predictions)

    count_predictions = len(predictions)
    count_species = len(classes)
    assert top_n <= count_species
    submission = []
    # convert float to int
    glc_ids = [int(g) for g in glc_ids]
    classes = [int(c) for c in classes]

    for i in tqdm(range(count_predictions)):
        species = list(classes)
        current_glc_id = glc_ids[i]
        current_predictions = predictions[i]
        # each prediction should contains probabilities for all species
        assert len(current_predictions) == count_species

        current_ranks = rankdata(current_predictions, method="ordinal")
        # convert rank 100,99,98 to rank 1,2,3
        current_ranks = count_species - current_ranks + 1
        current_glc_id_array = count_species * [current_glc_id]

        # sort after rank
        current_ranks, species, current_predictions = zip(*sorted(zip(current_ranks, species, current_predictions)))

        submissions = [list(a) for a in zip(current_glc_id_array, species, current_predictions, current_ranks)]

        # select only top_n ranks
        submissions = submissions[:top_n]
        submission.extend(submissions)
    
    return submission

def _get_df(submission):
    '''Returns a DataFrame for the created submission.'''
    submission_df = pd.DataFrame(submission, columns = ['patch_id', 'species_glc_id', 'probability', 'rank'])
    return submission_df

def make_submission_df(top_n, classes, predictions, glc_ids):
    '''Creates the submission DataFrame for the given parameters.'''
    submission = _make_submission(top_n, classes, predictions, glc_ids)
    return _get_df(submission)

def make_submission_groups_df(top_n, groups, predictions, glc_ids, groups_dict, props):
    '''Creates the submission DataFrame for the given group parameters.'''
    submission = _make_submission_groups(top_n, groups, predictions, glc_ids, groups_dict, props)
    return _get_df(submission)
