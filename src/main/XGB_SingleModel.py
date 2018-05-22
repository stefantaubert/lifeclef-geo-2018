import module_support_main
import pandas as pd
import numpy as np
import time
import data_paths_main as data_paths
from scipy.stats import rankdata
from sklearn.model_selection import train_test_split
import xgboost as xgb
import submission_maker
import settings_main as settings
import time
import main_preprocessing
import json
import pickle
import os
import get_ranks
import mrr
import matplotlib.pyplot as plt
import datetime
import Log
from sklearn.preprocessing import LabelEncoder
import metrics
# class XGBMrrEval():
#     def __init__(self, classes, y_valid):
#         self.classes = classes
#         self.y_valid = y_valid
#         self.class_count = len(self.classes)

#     def evalute(self, y_predicted, y_true):
#         print("evaluate")
#         glc = [x for x in range(len(y_predicted))]
#         subm = submission_maker._make_submission(self.class_count, self.classes, y_predicted, glc)
#         ranks = get_ranks.get_ranks(subm, self.y_valid, self.class_count)
#         mrr_score = mrr.mrr_score(ranks)
#         return ("mrr", mrr_score)

class top_k_accuracy():
    def __init__(self):
        pass
    
    def evaluate(self, y_predicted, y_true):
        print(y_predicted)
        print(y_true)


class Model():
    def save_after_it(self, env):
        print("Saving model of iteration", str(env.iteration))
        env.model.save_model(data_paths.xgb_model + str(env.iteration))

    def __init__(self):
        main_preprocessing.create_datasets()

        self.train_columns = [ 
            'chbio_1', 'chbio_2', 'chbio_3', 'chbio_4', 'chbio_5', 'chbio_6',
            'chbio_7', 'chbio_8', 'chbio_9', 'chbio_10', 'chbio_11', 'chbio_12',
            'chbio_13', 'chbio_14', 'chbio_15', 'chbio_16', 'chbio_17', 'chbio_18','chbio_19', 
            'etp', 'alti', 'awc_top', 'bs_top', 'cec_top', 'crusting', 'dgh', 'dimp', 'erodi', 'oc_top', 'pd_top', 'text',
            'proxi_eau_fast', 'clc', 'latitude', 'longitude'
        ]

        x_text = pd.read_csv(data_paths.train)
        self.x_test = pd.read_csv(data_paths.test)
        
        y = x_text["species_glc_id"]

        # species_count = np.load(data_paths.y_array).shape[1]
        self.species_map = np.unique(y)
        self.species_count = len(self.species_map)
        #np.save(data_paths.xgb_species_map, classes_)

        self.x_train, self.x_valid, self.y_train, self.y_valid = train_test_split(x_text, y, test_size=settings.train_val_split, random_state=settings.seed)
        
        #np.save(data_paths.xgb_glc_ids, x_valid["patch_id"])

        self.test_glc_ids = list(self.x_test['patch_id'])
        self.valid_glc_ids = list(self.x_valid['patch_id'])
        self.x_test = self.x_test[self.train_columns]
        self.x_train = self.x_train[self.train_columns]
        self.x_valid = self.x_valid[self.train_columns]
        
        # Die Parameter für XGBoost erstellen.
        self.params = {}
        self.params['updater'] = 'grow_gpu'
        self.params['base_score'] = 0.5
        self.params['booster'] = 'gbtree'
        self.params['objective'] = 'multi:softprob'
        self.params['max_depth'] = 3
        self.params['learning_rate'] = 0.1
        self.params['seed'] = 4242
        #params['colsample_bytree'] = 0.8 #um die 
        self.params['silent'] = 1
        self.params['eval_metric'] = 'merror'
        self.params['num_class'] = len(self.species_map) #=3336
        self.params['num_boost_round'] = 1
        self.params['early_stopping_rounds'] = 5
        # params['colsample_bylevel'] = 1
        # params['colsample_bytree'] = 1
        # params['gamma'] = 0
        # params['min_child_weight'] = 1
        # params['max_delta_step'] = 0
        # params['missing'] = None
        # params['reg_alpha'] = 0
        # params['reg_lambda'] = 1
        # params['scale_pos_weight'] = 1
        # params['subsample'] = 1
        self.params['predictor'] = 'gpu_predictor'
        self.params['tree_method'] = 'gpu_hist'
        #params['grow_policy'] = 'depthwise' #'lossguide'
        #params['max_leaves'] = 255


    def predict(self):       
        le = LabelEncoder().fit(self.y_train)
        training_labels = le.transform(self.y_train)
        validation_labels = le.transform(self.y_valid)

        # Datenmatrix für die Eingabedaten erstellen.
        #x_train.to_csv(data_paths.xgb_trainchached, index=False)
        #d_train = xgb.DMatrix(data_paths.xgb_trainchached + "#d_train.cache", label=training_labels)
        d_train = xgb.DMatrix(self.x_train, label=training_labels)
        d_valid = xgb.DMatrix(self.x_valid, label=validation_labels)

        print("Training model...")
        
        #evaluator = XGBMrrEval(classes_, y_valid)
        watchlist = [
            (d_train, 'train'), 
            (d_valid, 'validation'),
        ]
        
        #top3_acc = metrics.get_top3_accuracy()
        #top10_acc = metrics.get_top10_accuracy()
        #top50_acc = metrics.get_top50_accuracy()

        #xgb.callback.print_evaluation() 
        bst = xgb.train(self.params, d_train, num_boost_round=self.params["num_boost_round"], verbose_eval=None, feval=self.top_k_accuracy, evals=watchlist, early_stopping_rounds=self.params["early_stopping_rounds"])
        #bst = xgb.train(params, d_train, 1, verbose_eval=2, evals=watchlist, evaluator.evalute, callbacks=[self.save_after_it])

        print("Save model...")
        bst.save_model(data_paths.xgb_model)
        bst.dump_model(data_paths.xgb_model_dump)

        self.plt_features(bst, d_train)
        
        print("Predict validation data...")
        self.valid_predictions = bst.predict(d_valid)        

        print("Predict test data...")    
        d_test = xgb.DMatrix(self.x_test)
        self.test_predictions = bst.predict(d_test)        

    # def predict_test_set_from_saved_model(self, iteration_nr):
    #     print("Load model...")
    #     path = data_paths.xgb_model + str(iteration_nr)
    #     assert os.path.exists(path)

    #     bst = xgb.Booster(model_file=path)
    #     #bst.dump_model(data_paths.xgb_model_dump + str(iteration_nr))

    #     #testset = pd.read_csv(data_paths.test)
    #     #np.save(data_paths.xgb_test_glc_ids, testset["patch_id"])
        
    #     testset = testset[self.train_columns]
    #     testset_dmatrix = xgb.DMatrix(testset)
    #     #self.plt_features(bst, testset_dmatrix, iteration_nr)

    #     print("Predict test data...")    
    #     pred_test = bst.predict(testset_dmatrix)        

    #     print("Save test predictions...")
    #     np.save(data_paths.xgb_test_prediction, pred_test)

    def plt_features(self, bst, d_matrix):
        print("Plot feature importances...")
        # Ausschlagskraft aller Features plotten
        _, ax = plt.subplots(figsize=(12,18))
        # print("Features names:")
        # print(d_matrix.feature_names)
        # print("Fscore Items:")
        # print(bst.get_fscore().items())
        mapper = {'f{0}'.format(i): v for i, v in enumerate(d_matrix.feature_names)}
        mapped = {mapper[k]: v for k, v in bst.get_fscore().items()}
        xgb.plot_importance(mapped, color='red', ax=ax)
        #plt.show()
        plt.draw()
        plt.savefig(data_paths.xgb_feature_importances, bbox_inches='tight')
        print("Finished.", data_paths.xgb_feature_importances)

def run_without_groups():
    start_time = time.time()
    start_datetime = datetime.datetime.now()
    print("Start:", start_datetime)
    m = Model()
    print("Predict testset...")
    m.predict()
    print("Finished.")
    print("Create test submission...")
    df = submission_maker.make_submission_df(settings.TOP_N_SUBMISSION_RANKS, m.species_map, m.test_predictions, m.test_glc_ids)
    print("Save test submission...")
    df.to_csv(data_paths.xgb_singlemodel_submission, index=False, sep=";", header=None)
    print("Finished.", data_paths.xgb_singlemodel_submission)

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
    writeLog("XGBoost Single Model", start_datetime, end_date_time, duration_min, m, mrr_score)

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

if __name__ == "__main__":
    run_without_groups()
    