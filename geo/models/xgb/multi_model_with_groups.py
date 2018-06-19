'''
- extract all pixel information from the images to a csv file
- use average pixel value of each channel and round to 2 decimals (save as csv)
- used columns "latitude" and "longitude" from occurrences and merge all information to one csv
- calculate species groups with similar species:
    - remove species which occur < 10
    - look at the most common values for each species
    - calculate difference
    - take threshold '3' for building groups -> 2090 groups (366 species in groups with size greater than 1)
- predict each group with one model separately (2090 different models)
- each group contained several species which were ordered with regard of their probability in the trainset. After predicting a group on the testset we counted this prediction as predicting all classes of the group with descending probabilities.
Test-mrr with groups: 0.022000386535744
'''

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import multiprocessing as mp
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from joblib import Parallel, delayed
from tqdm import tqdm

from geo.models.settings import seed
from geo.models.settings import train_val_split
from geo.models.settings import TOP_N_SUBMISSION_RANKS
from geo.models.data_paths import xgb_multimodel_groups_submission
from geo.preprocessing.preprocessing import create_datasets
from geo.preprocessing.preprocessing import extract_groups
from geo.preprocessing.species_occurences import extract_species_occurences
from geo.preprocessing.species_occurences import load_species_occurences
from geo.data_paths import train_with_groups
from geo.data_paths import test
from geo.data_paths import named_groups
from geo.metrics.mrr import mrr_score
from geo.logging.log import log_start
from geo.logging.log import log_end_xgb
from geo.postprocessing.submission_maker import _make_submission_groups
from geo.postprocessing.submission_maker import make_submission_groups_df
from geo.postprocessing.get_ranks import get_ranks

train_columns = [ 
    'chbio_1', 'chbio_2', 'chbio_3', 'chbio_4', 'chbio_5', 'chbio_6',
    'chbio_7', 'chbio_8', 'chbio_9', 'chbio_10', 'chbio_11', 'chbio_12',
    'chbio_13', 'chbio_14', 'chbio_15', 'chbio_16', 'chbio_17', 'chbio_18','chbio_19', 
    'etp', 'alti', 'awc_top', 'bs_top', 'cec_top', 'crusting', 'dgh', 'dimp', 'erodi', 'oc_top', 'pd_top', 'text',
    'proxi_eau_fast', 'clc', 'latitude', 'longitude'
]

# setting the parameters for xgboost
params = {
    'objective': 'binary:logistic',
    'max_depth': 2,
    'seed': 4242,
    'silent': 1,
    'eval_metric': 'logloss', # because we want to evaluate floats
    'num_boost_round': 500,
    'early_stopping_rounds': 10,
    'verbose_eval': None,
    'updater': 'grow_gpu',
    'predictor': 'gpu_predictor',
    'tree_method': 'gpu_hist'
}

def run_multi_model_with_groups(use_multithread=True):
    log_start()
    print("Running xgboost multi model with groups...")
    create_datasets()
    extract_groups()
    x_text = pd.read_csv(train_with_groups)
    extract_species_occurences()
    species_occ = load_species_occurences()
    n_groups = np.load(named_groups)
    species_occ_dict = {}
    for _, row in species_occ.iterrows():
        species_occ_dict[row["species"]] = row["percents"]
        
    x_test = pd.read_csv(test)      
    y = x_text["species_glc_id"]

    class_names = np.unique(y)

    x_train, x_valid, y_train, y_valid = train_test_split(x_text, y, test_size=train_val_split, random_state=seed)
    test_glc_ids = list(x_test["patch_id"])
    valid_glc_ids = list(x_valid["patch_id"])
    x_train = x_train[train_columns]
    x_valid = x_valid[train_columns]
    x_test = x_test[train_columns]

    if use_multithread:
        num_cores = mp.cpu_count()
        print("Cpu count:", str(num_cores))
        result = Parallel(n_jobs=num_cores)(delayed(predict_species)(class_name,x_train, x_valid, x_test, y_train, y_valid) for class_name in tqdm(class_names))
    else:
        result = []
        for class_name in tqdm(class_names):
            result.append(predict_species(class_name, x_train, x_valid, x_test, y_train, y_valid))

    species = np.array([x for x, _, _ in result])
    #transpose because each species is a column
    predictions = np.array([y for _, y, _ in result]).T 
    test_predictions = np.array([z for _, _, z in result]).T

    species_map = species
    species_count = len(species_map)
    valid_predictions = predictions
    test_predictions = test_predictions

    assert len(valid_predictions) == len(y_valid.index)
    assert len(test_predictions) == len(x_test.index)
    assert len(valid_predictions[0]) == species_count
    assert len(test_predictions[0]) == species_count

    print("Create test submission...")    
    df = make_submission_groups_df(TOP_N_SUBMISSION_RANKS, species_map, test_predictions, test_glc_ids, n_groups, species_occ_dict)
    df.to_csv(xgb_multimodel_groups_submission, index=False, sep=";", header=None)
    print("Finished.", xgb_multimodel_groups_submission)

    print("Evaluate validation set...")    
    subm = _make_submission_groups(TOP_N_SUBMISSION_RANKS, species_map, valid_predictions, valid_glc_ids, n_groups, species_occ_dict)
    ranks = get_ranks(subm, y_valid, TOP_N_SUBMISSION_RANKS)
    score = mrr_score(ranks)
    print("MRR-Score:", score * 100, "%")
    log_end_xgb("XGBoost Multi Model With Groups", train_columns, params, score)

def predict_species(species, x_train, x_valid, x_test, y_train, y_valid):
    train_target = list(map(lambda x: 1 if x == species else 0, y_train))
    val_target = list(map(lambda x: 1 if x == species else 0, y_valid))
    d_train = xgb.DMatrix(x_train, label=train_target)
    d_valid = xgb.DMatrix(x_valid, label=val_target)
    d_test = xgb.DMatrix(x_test)
    watchlist = [(d_train, 'train'), (d_valid, 'valid')]
    
    bst = xgb.train(
        params, 
        d_train, 
        num_boost_round=params["num_boost_round"], 
        verbose_eval=params["verbose_eval"],
        evals=watchlist, 
        early_stopping_rounds=params["early_stopping_rounds"]
    )

    plt_features(bst, d_train)
    pred = bst.predict(d_valid, ntree_limit=bst.best_ntree_limit)
    #print("validation-logloss for", str(species) + ":", log_loss(val_target, pred))
    pred_test = bst.predict(d_test, ntree_limit=bst.best_ntree_limit)

    return (species, pred, pred_test)

def plt_features(bst, d_test):
    print("Plot feature importances...")
    _, ax = plt.subplots(figsize=(12,18))
    xgb.plot_importance(bst, color='red', ax=ax)
    plt.show()

if __name__ == '__main__':
    run_multi_model_with_groups(use_multithread=False)