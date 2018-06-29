# Documentation XGBoost
# http://xgboost.readthedocs.io/en/latest/parameter.html
# http://xgboost.readthedocs.io/en/latest/python/python_api.html
# http://xgboost.readthedocs.io/en/latest/gpu/index.html

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from geo.models.settings import seed
from geo.models.settings import train_val_split
from geo.models.settings import TOP_N_SUBMISSION_RANKS
from geo.models.data_paths import xgb_singlemodel_submission
from geo.models.data_paths import xgb_model
from geo.models.data_paths import xgb_model_dump
from geo.models.data_paths import xgb_feature_importances
from geo.preprocessing.preprocessing import create_datasets
from geo.data_paths import train
from geo.data_paths import test
from geo.metrics.mrr import mrr_score
from geo.metrics.top_k_acc import top_k_error_eval
from geo.logging.log import log_start
from geo.logging.log import log_end_xgb
from geo.postprocessing.submission_maker import _make_submission
from geo.postprocessing.submission_maker import make_submission_df
from geo.postprocessing.get_ranks import get_ranks

train_columns = [
    'chbio_1', 'chbio_2', 'chbio_3', 'chbio_4', 'chbio_5', 'chbio_6',
    'chbio_7', 'chbio_8', 'chbio_9', 'chbio_10', 'chbio_11', 'chbio_12',
    'chbio_13', 'chbio_14', 'chbio_15', 'chbio_16', 'chbio_17', 'chbio_18','chbio_19', 
    'etp', 'alti', 'awc_top', 'bs_top', 'cec_top', 'crusting', 'dgh', 'dimp', 'erodi', 'oc_top', 'pd_top', 'text',
    'proxi_eau_fast', 'clc', 'latitude', 'longitude'
]

def run_single_model():
    log_start()
    print("Running xgboost single model...")
    create_datasets()
    x_text = pd.read_csv(train)
    x_test = pd.read_csv(test)
    y = x_text["species_glc_id"]
    species_map = np.unique(y)
    species_count = len(species_map)

    x_train, x_valid, y_train, y_valid = train_test_split(x_text, y, test_size=train_val_split, random_state=seed)

    test_glc_ids = list(x_test['patch_id'])
    valid_glc_ids = list(x_valid['patch_id'])
    x_test = x_test[train_columns]
    x_train = x_train[train_columns]
    x_valid = x_valid[train_columns]

    # create data matrix for the datasets
    le = LabelEncoder().fit(y_train)
    training_labels = le.transform(y_train)
    validation_labels = le.transform(y_valid)
    d_train = xgb.DMatrix(x_train, label=training_labels)
    d_valid = xgb.DMatrix(x_valid, label=validation_labels)

    watchlist = [
        #(d_train, 'train'), 
        (d_valid, 'validation'),
    ]
            
    evaluator = top_k_error_eval(species_map, y_valid, k=20)
    # bst = xgb.Booster(model_file=path)
    
    # setting the parameters for xgboost
    params = {
        'objective': 'multi:softprob',
        'max_depth': 2,
        'seed': 4242,
        'silent': 0,
        'eval_metric': 'merror',
        'num_class': len(species_map),
        'num_boost_round': 180,
        'early_stopping_rounds': 10,
        'verbose_eval': 1,
        'updater': 'grow_gpu',
        'predictor': 'gpu_predictor',
        'tree_method': 'gpu_hist'
    }

    print("Training model...")
    bst = xgb.train(
        params,
        d_train, 
        num_boost_round=params["num_boost_round"], 
        verbose_eval=params["verbose_eval"],
        #feval=evaluator.evaluate, 
        evals=watchlist, 
        #early_stopping_rounds=params["early_stopping_rounds"]
        #callbacks=[save_after_it]
    )

    print("Save model...")
    bst.save_model(xgb_model)
    bst.dump_model(xgb_model_dump)

    #plt_features(bst, d_train)

    print("Predict test set and create submission...")    
    d_test = xgb.DMatrix(x_test)
    test_predictions = bst.predict(d_test, ntree_limit=bst.best_ntree_limit)        
    df = make_submission_df(TOP_N_SUBMISSION_RANKS, species_map, test_predictions, test_glc_ids)
    df.to_csv(xgb_singlemodel_submission, index=False, sep=";", header=None)
    print("Finished.", xgb_singlemodel_submission)

    print("Predict & evaluate validation set...")    
    valid_predictions = bst.predict(d_valid, ntree_limit=bst.best_ntree_limit)
    print(evaluator.evaluate(valid_predictions, y_valid))
    subm = _make_submission(species_count, species_map, valid_predictions, valid_glc_ids)
    ranks = get_ranks(subm, y_valid, species_count)
    score = mrr_score(ranks)
    print("MRR-Score:", score * 100, "%")
    log_end_xgb("XGBoost Single Model", train_columns, params, score)

def plt_features(bst, d_matrix):
    print("Plot feature importances...")
    _, ax = plt.subplots(figsize=(12,18))
    xgb.plot_importance(bst.get_fscore(), color='red', ax=ax)
    #plt.show()
    plt.draw()
    plt.savefig(xgb_feature_importances, bbox_inches='tight')
    print("Finished.", xgb_feature_importances)

def save_after_it(env):
    print("Saving model of iteration", str(env.iteration))
    env.model.save_model(xgb_model + str(env.iteration))

if __name__ == "__main__":
    run_single_model()
