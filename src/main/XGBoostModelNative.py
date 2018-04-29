import module_support_main
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
import get_ranks
import mrr
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

class XGBMrrEval():
    def __init__(self, classes, y_valid):
        self.classes = classes
        self.y_valid = y_valid
        self.class_count = len(self.classes)

    def evalute(self, y_predicted, y_true):
        print("evaluate")
        glc = [x for x in range(len(y_predicted))]
        subm = submission_maker._make_submission(self.class_count, self.classes, y_predicted, glc)
        ranks = get_ranks.get_ranks(subm, self.y_valid, self.class_count)
        mrr_score = mrr.mrr_score(ranks)
        return ("mrr", mrr_score)

class XGBModelNative():
    def save_after_it(self, env):
        print("Saving model of iteration", str(env.iteration))
        env.model.save_model(data_paths.xgb_model + str(env.iteration))

    def __init__(self):
        self.train_columns = [ 
        'alti', 'bs_top', 'chbio_12', 'chbio_15', 'chbio_17', 'chbio_3', 'chbio_6', 'clc', 'crusting', 'dimp'
        # 'chbio_1', 'chbio_2', 'chbio_3', 'chbio_4', 'chbio_5', 'chbio_6',
        # 'chbio_7', 'chbio_8', 'chbio_9', 'chbio_10', 'chbio_11', 'chbio_12',
        # 'chbio_13', 'chbio_14', 'chbio_15', 'chbio_16', 'chbio_17', 'chbio_18','chbio_19', 
        # 'etp', 'alti', 'awc_top', 'bs_top', 'cec_top', 'crusting', 'dgh', 'dimp', 'erodi', 'oc_top', 'pd_top', 'text',
        # 'proxi_eau_fast', 'clc', 'latitude', 'longitude'
        ]

    def run(self):
        print("Run model...")
        #x_text = np.load(data_paths.x_text)
        x_text = pd.read_csv(data_paths.train)
        y = x_text["species_glc_id"]

        # species_count = np.load(data_paths.y_array).shape[1]
        classes_ = np.unique(y)
        np.save(data_paths.xgb_species_map, classes_)

        x_train, x_valid, y_train, y_valid = train_test_split(x_text, y, test_size=settings.train_val_split, random_state=settings.seed)
        
        np.save(data_paths.xgb_glc_ids, x_valid["patch_id"])

        x_train = x_train[self.train_columns]
        x_valid = x_valid[self.train_columns]
        
        # Die Parameter für XGBoost erstellen.
        params = {}
        params['updater'] = 'grow_gpu'
        params['base_score'] = 0.5
        params['booster'] = 'gbtree'
        params['colsample_bylevel'] = 1
        params['colsample_bytree'] = 1
        params['gamma'] = 0
        params['max_depth'] = 10
        params['learning_rate'] = 0.1
        params['min_child_weight'] = 1
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
        #params['predictor'] = 'gpu_predictor'
        #params['tree_method'] = 'gpu_hist'
        #params['grow_policy'] = 'depthwise' #'lossguide'
        #params['max_leaves'] = 255

        le = LabelEncoder().fit(y_train)
        training_labels = le.transform(y_train)
        validation_labels = le.transform(y_valid)

        # Datenmatrix für die Eingabedaten erstellen.
        #x_train.to_csv(data_paths.xgb_trainchached, index=False)
        #d_train = xgb.DMatrix(data_paths.xgb_trainchached + "#d_train.cache", label=training_labels)
        d_train = xgb.DMatrix(x_train, label=training_labels)
        d_valid = xgb.DMatrix(x_valid, label=validation_labels)

        print("Training model...")
        
        evaluator = XGBMrrEval(classes_, y_valid)
        self.current_boosting_round = 0
        watchlist = [
            #(d_train, 'train'), 
            (d_valid, 'validation'),
        ]
        xgb.callback.print_evaluation() 
        bst = xgb.train(params, d_train, 1, verbose_eval=2, evals=watchlist, feval=evaluator.evalute, callbacks=[self.save_after_it])

        print("Save model...")
        bst.save_model(data_paths.xgb_model)
        bst.dump_model(data_paths.xgb_model_dump)

        # print("Predict validation data...")
        # test_dmatrix = xgb.DMatrix(x_valid)
        # pred = bst.predict(test_dmatrix)        

        # print("Save validation predictions...")
        # np.save(data_paths.xgb_prediction, pred)

    def predict_test_set_from_saved_model(self, iteration_nr):
        print("Load model...")
        path = data_paths.xgb_model + str(iteration_nr)
        assert os.path.exists(path)

        bst = xgb.Booster(model_file=path)
        #bst.dump_model(data_paths.xgb_model_dump + str(iteration_nr))

        testset = pd.read_csv(data_paths.test)
        np.save(data_paths.xgb_test_glc_ids, testset["patch_id"])
        
        testset = testset[self.train_columns]
        testset_dmatrix = xgb.DMatrix(testset)
        self.plt_features(bst, testset_dmatrix, iteration_nr)

        print("Predict test data...")    
        pred_test = bst.predict(testset_dmatrix)        

        print("Save test predictions...")
        np.save(data_paths.xgb_test_prediction, pred_test)

    def plt_features(self, bst, d_test, iteration_nr):
        print("Plot feature importances...")
        # Ausschlagskraft aller Features plotten
        _, ax = plt.subplots(figsize=(12,18))
        # print("Features names:")
        # print(d_test.feature_names)
        # print("Fscore Items:")
        # print(bst.get_fscore().items())
        mapper = {'f{0}'.format(i): v for i, v in enumerate(d_test.feature_names)}
        mapped = {mapper[k]: v for k, v in bst.get_fscore().items()}
        xgb.plot_importance(mapped, color='red', ax=ax)
        #plt.show()
        plt.draw()
        plt.savefig(data_paths.xgb_feature_importances, bbox_inches='tight')
        print("Finished.", data_paths.xgb_feature_importances)

if __name__ == "__main__":
    xg = XGBModelNative()
    #xg.run()
    xg.predict_test_set_from_saved_model(36)
    