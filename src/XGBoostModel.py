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
import DataReader
import settings
import time
import json

def run_Model():
    print("Run model...")
    #x_text = np.load(data_paths.x_text)

    x_text = pd.read_csv(data_paths.occurrences_train_gen)

    x_text = x_text[['chbio_1', 'chbio_5', 'chbio_6','month', 'latitude', 'longitude']]

    y = np.load(data_paths.y_ids)

    # species_count = np.load(data_paths.y_array).shape[1]
    
    x_train, x_valid, y_train, y_valid = train_test_split(x_text, y, test_size=settings.train_val_split, random_state=settings.seed)

    xg = XGBClassifier(
        objective="multi:softmax",
        eval_metric="merror",
        random_state=settings.seed,
        n_jobs=-1,
        n_estimators=30,
        # predictor='gpu_predictor',
    )

    if (True):
        # clf = DecisionTreeClassifier()
        # clf.n_classes_ = species_count
        # clf.fit(x_train, y_train)
        # pred = clf.predict_proba(x_valid)
        classes_ = np.unique(y)

        # Die Parameter für XGBoost erstellen.
        params = {}
        params['objective'] = 'multi:softmax'
        params['eval_metric'] = 'merror'
        params['eta'] = 0.02
        params['max_depth'] = 3
        params['subsample'] = 0.6
        params['base_score'] = 0.2
        params['num_class'] = len(classes_) + 1 # da species_id 1-basiert ist
        # params['scale_pos_weight'] = 0.36 #für test set

        # Berechnungen mit der GPU ausführen
        # params['updater'] = 'grow_gpu'

        # Datenmatrix für die Eingabedaten erstellen.
        d_train = xgb.DMatrix(x_train, label=y_train)
        d_valid = xgb.DMatrix(x_valid, label=y_valid)
        np.save(data_paths.species_map_training, classes_)

        # Um den Score für das Validierungs-Set während des Trainings zu berechnen, muss eine Watchlist angelegt werden.
        watchlist = [(d_train, 'train'), (d_valid, 'valid')]

        # Modell trainieren.
        # Geschwindigkeit ca. 1000 pro Minute auf der P6000
        # zeigt alle 10 Schritte den Score für das Validierungs-Set an
        print("Training model...")
        bst = xgb.train(params, d_train, 5, watchlist, verbose_eval=1)

        # Modell speichern.
        bst.dump_model(data_paths.model_dump)
        bst.save_model(data_paths.model)

        pred = bst.predict_proba(d_valid)
        np.save(data_paths.prediction, pred)
        
    else:
        print("Fit model...")
        xg.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_valid, y_valid)])
        np.save(data_paths.species_map_training, xg.classes_)

        print("Predict data...")
        pred = xg.predict_proba(x_valid)

        np.save(data_paths.prediction, pred)


if __name__ == '__main__':
    start_time = time.time()

    # DataReader.read_and_write_data()
    run_Model()
    submission_maker.make_submission()
    evaluation.evaluate_with_mrr()

    print("Total duration:", time.time() - start_time, "s")