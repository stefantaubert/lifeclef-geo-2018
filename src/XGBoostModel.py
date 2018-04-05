import pandas as pd
import numpy as np
import time
from scipy.stats import rankdata
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import data_paths
import submission_maker
import evaluation
import DataReader
import settings
import time

def run_Model():
    print("Run model...")
    x_text = np.load(data_paths.x_text)
    y = np.load(data_paths.y_ids)

    # species_count = np.load(data_paths.y_array).shape[1]
    
    x_train, x_valid, y_train, y_valid = train_test_split(x_text, y, test_size=settings.train_val_split, random_state=settings.seed)
   
    # Ist durch den RandomFinder behoben
    # # Entferne Spezies aus dem Validierungsset, falls diese Spezies nicht im Trainingsset vorkommt
    # indicies = []
    # species_only_in_val = []
    # for index in range(0, len(y_valid)):
    #     item = y_valid[index]
    #     if item not in y_train:
    #         indicies.append(index)
    #         species_only_in_val.append(item)
    # x_valid = np.delete(x_valid, indicies, 0)
    # y_valid = np.delete(y_valid, indicies)
    # print("Dropped indicies from validationset because they are not in trainingsset:", indicies, "Count:", len(indicies))
    
    xg = XGBClassifier(
        objective="multi:softmax",
        eval_metric="merror",
        random_state=settings.seed,
        n_jobs=-1,
        n_estimators=30,
        predictor='gpu_predictor',
    )

    print("Fit model...")
    xg.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_valid, y_valid)])

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