import pandas as pd
import numpy as np
import time
from scipy.stats import rankdata
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import xgboost as xgb
import data_paths
import submission_maker
import evaluation
import settings
import time
import json
import os

class XGBModel():
    def __init__(self):
        self.create_output_dir_if_not_exists()

    def create_output_dir_if_not_exists(self):
        if not os.path.exists(data_paths.xgb_dir):
            os.makedirs(data_paths.xgb_dir)

    def run(self):
        print("Run model...")
        #x_text = np.load(data_paths.x_text)

        x_text = pd.read_csv(data_paths.xgb_train_groups)
        y = x_text["species_glc_id"]
        train_columns = [ 'chbio_1', 'chbio_2', 'chbio_3', 'chbio_4', 'chbio_5', 'chbio_6',
        'chbio_7', 'chbio_8', 'chbio_9', 'chbio_10', 'chbio_11', 'chbio_12',
        'chbio_13', 'chbio_14', 'chbio_15', 'chbio_16', 'chbio_17', 'chbio_18',
        'chbio_19', 
        # 'etp', 'alti', 'awc_top', 'bs_top', 'cec_top', 'crusting', 'dgh', 'dimp', 'erodi', 'oc_top', 'pd_top', 'text',
        # 'proxi_eau_fast', 'clc', 'latitude', 'longitude'
        ]

        # species_count = np.load(data_paths.y_array).shape[1]
        
        x_train, x_valid, y_train, y_valid = train_test_split(x_text, y, test_size=settings.train_val_split, random_state=settings.seed)
        
        validation_glc_ids = x_valid["patch_id"]
        np.save(data_paths.xgb_glc_ids, validation_glc_ids)

        x_train = x_train[train_columns]
        x_valid = x_valid[train_columns]
        
        xg = XGBClassifier(
            objective="multi:softmax",
            eval_metric="merror",
            random_state=settings.seed,
            n_jobs=-1,
            n_estimators=1,
            predictor='gpu_predictor',
        )

        print("Fit model...")
        xg.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_valid, y_valid)])
        np.save(data_paths.xgb_group_map, xg.classes_)

        # print("Save model...")
        # xg.dump_model(data_paths.model_dump)
        # xg.save_model(data_paths.model)

        print("Predict data...")
        pred = xg.predict_proba(x_valid)

        print("Save predictions...")
        np.save(data_paths.xgb_prediction, pred)