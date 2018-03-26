import time
import pandas as pd
import numpy as np
import time
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import rankdata
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score, recall_score, precision_score, f1_score
from xgboost import XGBClassifier
from sklearn.multiclass import OneVsRestClassifier
from XGBClassifierWithParam import XGBCl

import data_paths

if __name__ == '__main__':
    all_start = time.time()

    x_train = pd.read_csv(data_paths.features_train, nrows=1000)
    print(x_train)
    species_count = len(set(x_train["species_glc_id"].values))

    # Ausgabedaten erstellen.
    y_train = x_train.species_glc_id
    x_train.drop(["species_glc_id"], axis=1, inplace=True)

    # Trainings-Set und Validierungs-Set erstellen. Das Validierungs-Set enthält 10% aller Trainings-Daten.
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=4242)
    print("Validationset rows:", len(x_valid.index))

    # Entferne Spezies aus dem Validierungsset, falls diese Spezies nicht im Trainingsset vorkommt
    indicies = []
    for index, item in y_valid.iteritems():
        if item not in y_train.values:
            indicies.append(index)

    x_valid.drop(indicies, inplace=True)
    y_valid.drop(indicies, inplace=True)

    print("Validationset rows after removing unique species:", len(x_valid.index))

    train_species_ids = list(set(y_train))
    val_species_ids = list(set(y_valid))

    # print("Trainspecies:", train_species_ids)
    print("Trainspecies count:", len(train_species_ids))
    # print("Trainspecies last:", list(train_species_ids)[-1])

    # print("Valspecies:", val_species_ids)
    print("Valspecies count:", len(val_species_ids))
    # print("Valspecies last:", list(val_species_ids)[-1])

    #x_train.drop(['patch_id'], axis=1)

    x_train_ids = x_train.glc_id
    x_train.drop(["glc_id"],axis=1, inplace=True)

    # print("Trainfeatures:", x_train)
    # print("Train glc_ids:", x_train_ids)
    print("Unknown species in trainset:", species_count - len(train_species_ids))

    x_valid_ids = list(x_valid.glc_id)
    x_valid.drop(["glc_id"],axis=1, inplace=True)

    # print("Validationfeatures:", x_valid)
    # print("Validation glc_ids:", x_valid_ids)
    # print("Count Validation glc_ids:", len(x_valid_ids))
    print("Species only in trainset:", len(set(val_species_ids).difference(set(train_species_ids))) )

    if (False):
        clf = DecisionTreeClassifier()
        clf.n_classes_ = species_count
        clf.fit(x_train, y_train)
        pred = clf.predict_proba(x_valid)
    else:
        xg = XGBClassifier(objective="multi:softmax", eval_metric="merror", random_state=4242, n_jobs=-1, n_estimators = 36)
        # xg.fit()

        # ctf = OneVsRestClassifier(xg, n_jobs=-1)
        # print(ctf)
        print("Fit model...")
        xg.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_valid, y_valid)])

        #ctf.fit(x_train, y_train)
        print("Predict data...")
        pred = xg.predict_proba(x_valid)

    print("Process data...")
    # print(pred)
    # print(pred.shape)

    #<glc_id;species_glc_id;probability;rank>

    result = pd.DataFrame(columns=['glc_id', 'species_glc_id', 'probability', 'rank', 'real_species_glc_id'])

    for i in range(0, len(pred)):
        current_pred = pred[i]
        current_glc_id = int(x_valid_ids[i])
        #pred_ranked = pd.Series(current_pred).rank(ascending=False, method="min")
        pred_r = rankdata(current_pred, method="ordinal")
        # absteigend sortieren
        pred_r = len(train_species_ids) - pred_r + 1
        glc_id_array = [int(current_glc_id)] * len(train_species_ids)
        #glc_id_array = pd.to_numeric(glc_id_array, downcast='integer')

        sol_array = [int(list(y_valid)[i])] * len(train_species_ids)

        percentile_list = pd.DataFrame({
            'glc_id': glc_id_array,
            'species_glc_id': train_species_ids,
            'probability': current_pred,
            'rank': pred_r,
            'real_species_glc_id': sol_array,
        })

        # macht aus int float!
        #percentile_list = pd.DataFrame(np.column_stack([glc_id_array, train_species_ids, current_pred, pred_r]), columns=['glc_id', 'species_glc_id', 'probability', 'rank'])
        result = pd.concat([result, percentile_list], ignore_index=True)
    result = result.reindex(columns=('glc_id', 'species_glc_id', 'probability', 'rank', 'real_species_glc_id'))

    result = result[result.species_glc_id == result.real_species_glc_id]

    #print(result)

    print("Calculate MRR-Score...")
    sum = 0.0
    Q = len(result.index)
    print("Q:", Q)

    print("Dropped valpredictions because species was only in valset:", len(y_valid.index) - Q)

    # MRR berechnen
    for index, row in result.iterrows():
        sum += 1 / float(row["rank"])

    mrr_score = 1.0 / Q * sum
    print("MRR-Score:", mrr_score)

    result.drop(['real_species_glc_id'],axis=1, inplace=True)
    result.to_csv(data_paths.submission_val, index=False, sep=";")
    print("Total duration (s):", time.time() - all_start)