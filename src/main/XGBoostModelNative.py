import pandas as pd
import numpy as np
import time
from scipy.stats import rankdata
from sklearn.model_selection import train_test_split
#from xgboost import XGBClassifier
import xgboost as xgb
import data_paths_main as data_paths
import submission_maker
import evaluation
import settings_main as settings
import time
import json
import pickle
import os

from sklearn.preprocessing import LabelEncoder

class XGBModelNative():
    def mrr_eval(self, y_predicted, y_true):
        print(y_predicted, y_true)
        return ("test", 0.5)

    def run(self):
        print("Run model...")
        #x_text = np.load(data_paths.x_text)

        x_text = pd.read_csv(data_paths.train)
        y = x_text["species_glc_id"]
        train_columns = [ 
        'alti', 'bs_top', 'chbio_12', 'chbio_15', 'chbio_17', 'chbio_3', 'chbio_6', 'clc', 'crusting', 'dimp'
        # 'chbio_1', 'chbio_2', 'chbio_3', 'chbio_4', 'chbio_5', 'chbio_6',
        # 'chbio_7', 'chbio_8', 'chbio_9', 'chbio_10', 'chbio_11', 'chbio_12',
        # 'chbio_13', 'chbio_14', 'chbio_15', 'chbio_16', 'chbio_17', 'chbio_18','chbio_19', 
        # 'etp', 'alti', 'awc_top', 'bs_top', 'cec_top', 'crusting', 'dgh', 'dimp', 'erodi', 'oc_top', 'pd_top', 'text',
        # 'proxi_eau_fast', 'clc', 'latitude', 'longitude'
        ]

        # species_count = np.load(data_paths.y_array).shape[1]
        classes_ = np.unique(y)
        np.save(data_paths.xgb_species_map, classes_)

        x_train, x_valid, y_train, y_valid = train_test_split(x_text, y, test_size=settings.train_val_split, random_state=settings.seed)
        
        np.save(data_paths.xgb_glc_ids, x_valid["patch_id"])

        x_train = x_train[train_columns]
        x_valid = x_valid[train_columns]
        
        # Die Parameter für XGBoost erstellen.
        params = {}
        params['base_score'] = 0.5
        params['booster'] = 'gbtree'
        params['colsample_bylevel'] = 1
        params['colsample_bytree'] = 1
        params['gamma'] = 0
        params['learning_rate'] = 0.1
        params['max_delta_step'] = 0
        params['missing'] = None
        params['objective'] = 'multi:softprob'
        params['reg_alpha'] = 0
        params['reg_lambda'] = 1
        params['scale_pos_weight'] = 1
        params['seed'] = 4
        params['silent'] = 1
        params['subsample'] = 1
        params['eval_metric'] = 'merror'
        params['num_class'] = len(classes_) #3336
        params['updater'] = 'grow_gpu'
        params['predictor'] = 'gpu_predictor'
        params['tree_method'] = 'gpu_hist'
        params['grow_policy'] = 'lossguide'#'depthwise'
        params['max_depth'] = 0 #0
        params['min_child_weight'] = 1
        params['max_leaves'] = 255

        # +1 because error:=label must be in [0, num_class), num_class=3336 but found 3336 in label.

        # Berechnungen mit der GPU ausführen

        le = LabelEncoder().fit(y_train)
        training_labels = le.transform(y_train)
        validation_labels = le.transform(y_valid)

        # Datenmatrix für die Eingabedaten erstellen.
        #x_train.to_csv(data_paths.xgb_trainchached, index=False)
        #d_train = xgb.DMatrix(data_paths.xgb_trainchached + "#d_train.cache", label=training_labels)
        d_train = xgb.DMatrix(x_train, label=training_labels)
        d_valid = xgb.DMatrix(x_valid, label=validation_labels)

        # # Um den Score für das Validierungs-Set während des Trainings zu berechnen, muss eine Watchlist angelegt werden.
        # watchlist =[(x_train, y_train), (x_valid, y_valid)]
        # evals = list(
        #                 xgb.DMatrix(x[0], label=le.transform(x[1]))
        #                 for x in watchlist
        #             )
        # nevals = len(evals)
        # eval_names = ["validation_{}".format(i) for i in range(nevals)]
        # evals = list(zip(evals, eval_names))

        print("Training model...")
        bst = xgb.train(params, d_train, 10, verbose_eval=1, evals=[(d_train, 'train'), (d_valid, 'validation')])

        print("Save model...")
        bst.save_model(data_paths.xgb_model)
        bst.dump_model(data_paths.xgb_model_dump)

        print("Predict validation data...")
        test_dmatrix = xgb.DMatrix(x_valid)
        pred = bst.predict(test_dmatrix)        

        print("Save validation predictions...")
        np.save(data_paths.xgb_prediction, pred)

if __name__ == "__main__":
    XGBModelNative().run()