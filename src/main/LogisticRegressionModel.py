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
    def __init__(self):
        main_preprocessing.create_datasets() 
        x_text = pd.read_csv(data_paths.train)
        self.x_test = pd.read_csv(data_paths.test)
        y = x_text["species_glc_id"]
        self.train_columns = [ 
        #'bs_top', 'alti', 'chbio_12', 'chbio_15', 'chbio_17', 'chbio_3', 'chbio_6', 'clc', 'crusting', 'dimp'
        'chbio_1',
         'chbio_2', 'chbio_3', 'chbio_4', 'chbio_5', 'chbio_6',
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
        self.x_test = self.x_test[self.train_columns]

    def run(self):
        start_time = time.time()
        start_datetime = datetime.datetime.now().time()
        print("Start:", start_datetime)
        print("Run model...")

        num_cores = multiprocessing.cpu_count()
        result = []
        for class_name in tqdm(self.class_names):
            result.append(self.calc_class_xg(class_name))

        #result = Parallel(n_jobs=num_cores)(delayed(self.calc_class)(class_name) for class_name in tqdm(self.class_names))
        result = Parallel(n_jobs=num_cores)(delayed(self.calc_class_xg)(class_name) for class_name in tqdm(self.class_names))
        species = np.array([x for x, _, _ in result])
        predictions = np.array([y for _, y, _ in result])
        test_predictions = np.array([z for _, _, z in result])
        # print(species)
        # print(predictions)
        print("Finished.")
        print("Saving results...")
        np.save(data_paths.regression_species, species)
        np.save(data_paths.regression_prediction, predictions)
        np.save(data_paths.regression_test_prediction, test_predictions)
        print("Saving completed", data_paths.regression_species, data_paths.regression_prediction, data_paths.regression_test_prediction)

        assert len(predictions) == len(self.class_names)
        assert len(predictions[0]) == len(self.x_valid.index)
        result = predictions.T
        #print(result)
        mrr = self.evalute(result, self.y_valid, species)
        print("Validation MRR score:", str(mrr))
        #print('Total ACC score is {}'.format(np.mean(self.scores)))
        end_date_time = datetime.datetime.now().time()
        print("End:", end_date_time)
        seconds = time.time() - start_time
        duration_min = round(seconds / 60, 2)
        print("Total duration:", duration_min, "min")
        Log.write("XGBoost Model\nScore: {}\nStarted: {}\nFinished: {}\nDuration: {}min\nSuffix: {}\nTraincolumns: {}\n==========".format(
            str(mrr), 
            str(start_datetime), 
            str(end_date_time),
            str(duration_min),
            data_paths.get_suffix_pro(),
            ", ".join(self.train_columns)))
    
    def eval_from_files(self):
        species = np.load(data_paths.regression_species)
        predictions = np.load(data_paths.regression_prediction)
        result = predictions.T
        print(self.evalute(result, self.y_valid, species))

    def calc_class(self, class_name):
        train_target = list(map(lambda x: 1 if x == class_name else 0, self.y_train))
        #val_target = list(map(lambda x: 1 if x == class_name else 0, self.y_valid))
        #print(train_target)
        classifier = LogisticRegression(C=10, solver='sag', n_jobs=-1, random_state=settings.seed, max_iter=100)

        #cv_score = np.mean(cross_val_score(classifier, x_train, train_target, cv=3, scoring='roc_auc'))
        #scores.append(cv_score)
        #print('CV score for class {} is {}'.format(class_name, cv_score))
        classifier.fit(self.x_train, train_target)
        pred = classifier.predict_proba(self.x_valid)
        #print(pred)
        pred_real = pred[:, 1] # second is for class is 1
        #print("acc", accuracy_score(val_target, pred_real.round()))
        #score = log_loss(val_target, pred_real)
        #print('Score for class {} is {}'.format(class_name, score.round()))
        #self.scores.append(score)
        return (class_name, pred_real)

    def calc_class_xg(self, class_name):
        train_target = list(map(lambda x: 1 if x == class_name else 0, self.y_train))
        val_target = list(map(lambda x: 1 if x == class_name else 0, self.y_valid))
        #print(train_target)
        params = {}
        params['updater'] = 'grow_gpu'
        params['objective'] = 'binary:logistic'
        params['max_depth'] = 3
        params['learning_rate'] = 0.1
        params['seed'] = 4242
        params['silent'] = 1
        params['eval_metric'] = 'logloss'
        params['predictor'] = 'gpu_predictor'
        params['tree_method'] = 'gpu_hist'
        # Datenmatrix für die Eingabedaten erstellen.
        d_train = xgb.DMatrix(self.x_train, label=train_target)
        d_valid = xgb.DMatrix(self.x_valid, label=val_target)
        d_test = xgb.DMatrix(self.x_test)

        # Um den Score für das Validierungs-Set während des Trainings zu berechnen, muss eine Watchlist angelegt werden.
        watchlist = [(d_train, 'train'), (d_valid, 'valid')]
    
        bst = xgb.train(params, d_train, 300, verbose_eval=20, evals=watchlist, early_stopping_rounds=5)
        
        self.plt_features(bst, d_train)

        pred = bst.predict(d_valid)
        print("validation-logloss for", str(class_name) + ":", log_loss(val_target, pred))

        pred_test = bst.predict(d_test)
        
        #maximum = np.amax(pred)
        return (class_name, pred, pred_test)

    def plt_features(self, bst, d_test):
        print("Plot feature importances...")
        # Ausschlagskraft aller Features plotten
        _, ax = plt.subplots(figsize=(12,18))
        # print("Features names:")
        # print(d_test.feature_names)
        # print("Fscore Items:")
        # print(bst.get_fscore().items())
        # mapper = {'f{0}'.format(i): v for i, v in enumerate(d_test.feature_names)}
        # mapped = {mapper[k]: v for k, v in bst.get_fscore().items()}
        xgb.plot_importance(bst, color='red', ax=ax)
        plt.show()
        # plt.draw()
        # plt.savefig(data_paths.xgb_feature_importances, bbox_inches='tight')
        # print("Finished.", data_paths.xgb_feature_importances)

    def evalute(self, y_predicted, y_true, classes):
        print("evaluate")
        glc = [x for x in range(len(y_predicted))]
        subm = submission_maker._make_submission(len(classes), classes, y_predicted, glc)
        ranks = get_ranks.get_ranks(subm, y_true, len(classes))
        mrr_score = mrr.mrr_score(ranks)
        return mrr_score

if __name__ == '__main__':
    m = Model()
    #m.eval_from_files()
    m.run()