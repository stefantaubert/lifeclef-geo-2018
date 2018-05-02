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
import multiprocessing
import submission_maker
import get_ranks
import mrr
import time
import main_preprocessing
import Log
import datetime
import xgboost as xgb
import matplotlib.pyplot as plt

class Model():
    def __init__(self, use_groups = False):
        if use_groups:
            x_text = pd.read_csv(data_paths.train_with_groups)
        else:
            x_text = pd.read_csv(data_paths.train)
            
        x_test = pd.read_csv(data_paths.test)
        
        y = x_text["species_glc_id"]
        self.train_columns = [ 
        #'bs_top', 'alti', 'chbio_12', 'chbio_15', 'chbio_17', 'chbio_3', 'chbio_6', 'clc', 'crusting', 'dimp'
        'chbio_1', 'chbio_2', 'chbio_3', 'chbio_4', 'chbio_5', 'chbio_6',
        'chbio_7', 'chbio_8', 'chbio_9', 'chbio_10', 'chbio_11', 'chbio_12',
        'chbio_13', 'chbio_14', 'chbio_15', 'chbio_16', 'chbio_17', 'chbio_18','chbio_19', 
        'etp', 'alti', 'awc_top', 'bs_top', 'cec_top', 'crusting', 'dgh', 'dimp', 'erodi', 'oc_top', 'pd_top', 'text',
        'proxi_eau_fast', 'clc', 'latitude', 'longitude'
        ]
        # species_count = np.load(data_paths.y_array).shape[1]
        self.class_names = np.unique(y)
        #np.save(data_paths.xgb_species_map, classes_)

        self.x_train, self.x_valid, self.y_train, self.y_valid = train_test_split(x_text, y, test_size=settings.train_val_split, random_state=settings.seed)

        np.save(data_paths.xgb_species_map, self.class_names)
        np.save(data_paths.xgb_glc_ids, self.x_valid["patch_id"])

        self.x_train = self.x_train[self.train_columns]
        self.x_valid = self.x_valid[self.train_columns]
        self.x_test = x_test[self.train_columns]

        self.params = {}
        self.params['objective'] = 'binary:logistic'
        self.params['max_depth'] = 4
        self.params['learning_rate'] = 0.1
        self.params['seed'] = 4242
        self.params['silent'] = 1
        self.params['eval_metric'] = 'logloss'
        self.params['updater'] = 'grow_gpu'
        self.params['predictor'] = 'gpu_predictor'
        self.params['tree_method'] = 'gpu_hist'
        self.params['num_boost_round'] = 300
        self.params['early_stopping_rounds'] = 5

    def run(self, use_multithread = True):
        print("Run model...")
        if use_multithread:
            num_cores = multiprocessing.cpu_count()
            result = Parallel(n_jobs=num_cores)(delayed(self.calc_class_xg)(class_name) for class_name in tqdm(self.class_names))
            #result = Parallel(n_jobs=num_cores)(delayed(self.calc_class)(class_name) for class_name in tqdm(self.class_names))
        else:
            result = []
            for class_name in tqdm(self.class_names):
                result.append(self.calc_class_xg(class_name))

        species = np.array([x for x, _, _ in result])
        predictions = np.array([y for _, y, _ in result]).T #T weil jede species eine Spalte ist
        test_predictions = np.array([z for _, _, z in result]).T
        print("Finished.")

        print("Saving results...")
        np.save(data_paths.regression_species, species)
        np.save(data_paths.regression_prediction, predictions)
        np.save(data_paths.regression_test_prediction, test_predictions)
        print("Saving completed", data_paths.regression_species, data_paths.regression_prediction, data_paths.regression_test_prediction)

    def calc_class_xg(self, class_name):
        train_target = list(map(lambda x: 1 if x == class_name else 0, self.y_train))
        val_target = list(map(lambda x: 1 if x == class_name else 0, self.y_valid))
        d_train = xgb.DMatrix(self.x_train, label=train_target)
        d_valid = xgb.DMatrix(self.x_valid, label=val_target)
        d_test = xgb.DMatrix(self.x_test)
        watchlist = [(d_train, 'train'), (d_valid, 'valid')]
        
        bst = xgb.train(self.params, d_train, num_boost_round=self.params["num_boost_round"], verbose_eval=None, evals=watchlist, early_stopping_rounds=self.params["early_stopping_rounds"])
        #self.plt_features(bst, d_train)
        pred = bst.predict(d_valid)
        #print("validation-logloss for", str(class_name) + ":", log_loss(val_target, pred))
        pred_test = bst.predict(d_test)

        return (class_name, pred, pred_test)

    def plt_features(self, bst, d_test):
        print("Plot feature importances...")
        _, ax = plt.subplots(figsize=(12,18))
        xgb.plot_importance(bst, color='red', ax=ax)
        plt.show()

if __name__ == '__main__':
    m = Model(True)
    #m.eval_from_files()
    m.run(False)


    # def calc_class(self, class_name):
    #     train_target = list(map(lambda x: 1 if x == class_name else 0, self.y_train))
    #     #val_target = list(map(lambda x: 1 if x == class_name else 0, self.y_valid))
    #     #print(train_target)
    #     classifier = LogisticRegression(C=10, solver='sag', n_jobs=-1, random_state=settings.seed, max_iter=100)
    #     #cv_score = np.mean(cross_val_score(classifier, x_train, train_target, cv=3, scoring='roc_auc'))
    #     #scores.append(cv_score)
    #     #print('CV score for class {} is {}'.format(class_name, cv_score))
    #     classifier.fit(self.x_train, train_target)
    #     pred = classifier.predict_proba(self.x_valid)
    #     #print(pred)
    #     pred_real = pred[:, 1] # second is for class is 1
    #     #print("acc", accuracy_score(val_target, pred_real.round()))
    #     #score = log_loss(val_target, pred_real)
    #     #print('Score for class {} is {}'.format(class_name, score.round()))
    #     #self.scores.append(score)
    #     return (class_name, pred_real)