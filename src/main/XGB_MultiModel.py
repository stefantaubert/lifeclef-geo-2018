import module_support_main
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import data_paths_main as data_paths
import settings_main as settings
from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import cross_val_score
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
from scipy.sparse import hstack
from joblib import Parallel, delayed
import multiprocessing as mp
import submission_maker
import get_ranks
import mrr
import time
import main_preprocessing
import Log
import datetime
import xgboost as xgb
import matplotlib.pyplot as plt
import data_paths_analysis
import SpeciesOccurences

#http://xgboost.readthedocs.io/en/latest/parameter.html
#http://xgboost.readthedocs.io/en/latest/python/python_api.html
#http://xgboost.readthedocs.io/en/latest/gpu/index.html

class Model():
    '''
    No Group Model
    - extract all pixel information from the images to a csv file
    - use average pixel value of each channel and round to 12 decimals
    - used columns "latitude" and "longitude" from occurrences and merge all information to one csv
    - split trainset at 0.1 for validation set
    - run XGBoost with all feature columns (35)
    - predict each species with one model separately (3336 different models)
    - training with logloss and early stopping rounds
    Test-mrr without groups: 0.0338103315273545

    Group Model
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
    def __init__(self, use_groups):
        main_preprocessing.create_datasets()
        
        if use_groups:
            main_preprocessing.extract_groups()
            x_text = pd.read_csv(data_paths.train_with_groups)#, nrows=1000)
            SpeciesOccurences.create()
            species_occ = pd.read_csv(data_paths_analysis.species_occurences)
            self.named_groups = np.load(data_paths.named_groups)
            self.species_occ_dict = {}
            for _, row in species_occ.iterrows():
                self.species_occ_dict[row["species"]] = row["percents"]
        else:
            x_text = pd.read_csv(data_paths.train)#, nrows=1000)
            
        x_test = pd.read_csv(data_paths.test)#, nrows=1000)        
        y = x_text["species_glc_id"]

        self.train_columns = [ 
        'chbio_1', 'chbio_2', 'chbio_3', 'chbio_4', 'chbio_5', 'chbio_6',
        'chbio_7', 'chbio_8', 'chbio_9', 'chbio_10', 'chbio_11', 'chbio_12',
        'chbio_13', 'chbio_14', 'chbio_15', 'chbio_16', 'chbio_17', 'chbio_18','chbio_19', 
        'etp', 'alti', 'awc_top', 'bs_top', 'cec_top', 'crusting', 'dgh', 'dimp', 'erodi', 'oc_top', 'pd_top', 'text',
        'proxi_eau_fast', 'clc', 'latitude', 'longitude'
        ]

        self.class_names = np.unique(y)

        self.x_train, self.x_valid, self.y_train, self.y_valid = train_test_split(x_text, y, test_size=settings.train_val_split, random_state=settings.seed)
        self.test_glc_ids = list(x_test["patch_id"])
        self.valid_glc_ids = list(self.x_valid["patch_id"])
        self.x_train = self.x_train[self.train_columns]
        self.x_valid = self.x_valid[self.train_columns]
        self.x_test = x_test[self.train_columns]

        self.params = {}
        self.params['objective'] = 'binary:logistic'
        self.params['max_depth'] = 2
        self.params['learning_rate'] = 0.1
        self.params['seed'] = 4242
        self.params['silent'] = 1
        self.params['eval_metric'] = 'logloss' # because we want to evaluate floats
        self.params['updater'] = 'grow_gpu'
        self.params['predictor'] = 'gpu_predictor'
        self.params['tree_method'] = 'gpu_hist'
        self.params['num_boost_round'] = 500
        self.params['early_stopping_rounds'] = 10

    def predict(self, use_multithread):
        if use_multithread:
            num_cores = mp.cpu_count()
            print("Cpu count:", str(num_cores))
            result = Parallel(n_jobs=num_cores)(delayed(self.predict_species)(class_name) for class_name in tqdm(self.class_names))
        else:
            result = []
            for class_name in tqdm(self.class_names):
                result.append(self.predict_species(class_name))

        species = np.array([x for x, _, _ in result])
        #transpose because each species is a column
        predictions = np.array([y for _, y, _ in result]).T 
        test_predictions = np.array([z for _, _, z in result]).T

        self.species_map = species
        self.species_count = len(self.species_map)
        self.valid_predictions = predictions
        self.test_predictions = test_predictions

        assert len(self.valid_predictions) == len(self.y_valid.index)
        assert len(self.test_predictions) == len(self.x_test.index)
        assert len(self.valid_predictions[0]) == self.species_count
        assert len(self.test_predictions[0]) == self.species_count
    
    def predict_species(self, species):
        train_target = list(map(lambda x: 1 if x == species else 0, self.y_train))
        val_target = list(map(lambda x: 1 if x == species else 0, self.y_valid))
        d_train = xgb.DMatrix(self.x_train, label=train_target)
        d_valid = xgb.DMatrix(self.x_valid, label=val_target)
        d_test = xgb.DMatrix(self.x_test)
        watchlist = [(d_train, 'train'), (d_valid, 'valid')]
        
        bst = xgb.train(
            self.params, 
            d_train, 
            num_boost_round=self.params["num_boost_round"], 
            verbose_eval=None,
            evals=watchlist, 
            early_stopping_rounds=self.params["early_stopping_rounds"]
        )
        #self.plt_features(bst, d_train)
        pred = bst.predict(d_valid, ntree_limit=bst.best_ntree_limit)
        #print("validation-logloss for", str(species) + ":", log_loss(val_target, pred))
        pred_test = bst.predict(d_test, ntree_limit=bst.best_ntree_limit)

        return (species, pred, pred_test)

    def plt_features(self, bst, d_test):
        print("Plot feature importances...")
        _, ax = plt.subplots(figsize=(12,18))
        xgb.plot_importance(bst, color='red', ax=ax)
        plt.show()

def run_without_groups():
    start_time = time.time()
    start_datetime = datetime.datetime.now()
    print("Start:", start_datetime)
    m = Model(use_groups=False)
    print("Predict testset...")
    m.predict(use_multithread=True)
    print("Finished.")
    print("Create test submission...")
    df = submission_maker.make_submission_df(settings.TOP_N_SUBMISSION_RANKS, m.species_map, m.test_predictions, m.test_glc_ids)
    print("Save test submission...")
    df.to_csv(data_paths.xgb_multimodel_submission, index=False, sep=";", header=None)
    print("Finished.", data_paths.xgb_multimodel_submission)

    print("Evaluate submission...")
    print("Create valid submission...")
    subm = submission_maker._make_submission(m.species_count, m.species_map, m.valid_predictions, m.valid_glc_ids)
    print("Calculate score...")
    ranks = get_ranks.get_ranks(subm, m.y_valid, m.species_count)
    mrr_score = mrr.mrr_score(ranks)
    print("MRR-Score:", mrr_score * 100, "%")

    end_date_time = datetime.datetime.now()
    print("End:", end_date_time)
    seconds = time.time() - start_time
    duration_min = round(seconds / 60, 2)
    print("Total duration:", duration_min, "min")
    writeLog("XGBoost Multi Model", start_datetime, end_date_time, duration_min, m, mrr_score)

def run_with_groups():
    start_time = time.time()
    start_datetime = datetime.datetime.now()
    print("Start:", start_datetime)
    m = Model(use_groups=True)
    print("Predict testset...")
    m.predict(use_multithread=True)
    print("Finished.")
    print("Create test submission for groups...")
    # m.species_map are the groups in this case
    df = submission_maker.make_submission_groups_df(settings.TOP_N_SUBMISSION_RANKS, m.species_map, m.test_predictions, m.test_glc_ids, m.named_groups, m.species_occ_dict)
    print("Save test submission for groups...")
    df.to_csv(data_paths.xgb_multimodel_groups_submission, index=False, sep=";", header=None)
    print("Finished.", data_paths.xgb_multimodel_groups_submission)
    print("Evaluate submission...")
    print("Create valid submission...")
    subm = submission_maker._make_submission_groups(settings.TOP_N_SUBMISSION_RANKS, m.species_map, m.valid_predictions, m.valid_glc_ids, m.named_groups, m.species_occ_dict)
    ranks = get_ranks.get_ranks(subm, m.y_valid, settings.TOP_N_SUBMISSION_RANKS)
    mrr_score = mrr.mrr_score(ranks)
    print("MRR-Score:", mrr_score * 100,"%")
    end_date_time = datetime.datetime.now()
    print("End:", end_date_time)
    seconds = time.time() - start_time
    duration_min = round(seconds / 60, 2)
    print("Total duration:", duration_min, "min")
    writeLog("XGBoost Multi Model with groups", start_datetime, end_date_time, duration_min, m, mrr_score)

def writeLog(title, start, end, duration, model, mrr):
    log_text = str("{}\n--------------------\nMRR-Score: {}\nStarted: {}\nFinished: {}\nDuration: {}min\nSuffix: {}\nTraincolumns: {}\nSeed: {}\nSplit: {}\n".format
    (
        title,
        str(mrr), 
        str(start), 
        str(end),
        str(duration),
        data_paths.get_suffix_prot(),
        ", ".join(model.train_columns),
        settings.seed,
        settings.train_val_split,
    ))
    log_text += "Modelparams:\n"
    params = ["- {}: {}\n".format(x, y) for x, y in model.params.items()]
    log_text += "".join(params) + "============================="
    Log.write(log_text)
    print(log_text)

def run(use_groups):
    if use_groups:
        run_with_groups()
    else:
        run_without_groups()

if __name__ == '__main__':
    run(use_groups=True)