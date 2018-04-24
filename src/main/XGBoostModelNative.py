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

class XGBModel():
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
        x_train, x_valid, y_train, y_valid = train_test_split(x_text, y, test_size=settings.train_val_split, random_state=settings.seed)
        
        np.save(data_paths.xgb_glc_ids, x_valid["patch_id"])

        x_train = x_train[train_columns]
        x_valid = x_valid[train_columns]
        
        # Die Parameter für XGBoost erstellen.
        params = {}
        params['objective'] = 'multi:softmax'
        params['eval_metric'] = 'merror'
        # params['eta'] = 0.02
        # params['max_depth'] = 3
        # params['subsample'] = 0.6
        # params['base_score'] = 0.2
        params['num_class'] = len(classes_) + 1 # da species_id 1-basiert ist
        # params['scale_pos_weight'] = 0.36 #für test set

        # Berechnungen mit der GPU ausführen
        params['updater'] = 'grow_gpu'

        # Datenmatrix für die Eingabedaten erstellen.
        d_train = xgb.DMatrix(x_train, label=y_train)
        d_valid = xgb.DMatrix(x_valid, label=y_valid)

        # Um den Score für das Validierungs-Set während des Trainings zu berechnen, muss eine Watchlist angelegt werden.
        watchlist = [(d_train, 'train'), (d_valid, 'valid')]

        print("Training model...")
        bst = xgb.train(params, d_train, 2, watchlist, verbose_eval=1)

        print("Predict validation data...")
        test_dmatrix = xgb.DMatrix(x_valid)
        class_probs = bst.predict(test_dmatrix,output_margin=False,ntree_limit=0)
        classone_probs = class_probs
        classzero_probs = 1.0 - classone_probs
        pred = np.vstack((classzero_probs, classone_probs)).transpose()
    
        # print("Save model...")
        # pickle.dump(xg, open(data_paths.xgb_model, "wb"))
        # np.save(data_paths.xgb_species_map, xg.classes_)
        
        print("Save validation predictions...")
        np.save(data_paths.xgb_prediction, pred)