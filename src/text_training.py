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
import xgboost as xgb
from sklearn.multiclass import OneVsRestClassifier
from XGBClassifierWithParam import XGBCl
import data_paths

all_start = time.time()

x_train = np.load(data_paths.x_text)
y_train = np.load(data_paths.y)

print(x_train)
print(y_train)

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=4242)
print("Validationset rows:", len(x_valid))
print(x_valid)

# Entferne Spezies aus dem Validierungsset, falls diese Spezies nicht im Trainingsset vorkommt
indicies = []
for index in range(0, len(y_valid)):
    item = y_valid[index]
    if item not in y_train:
        indicies.append(index)

x_valid = np.delete(x_valid, indicies, 0)
y_valid = np.delete(y_valid, indicies)

train_species_ids = list(set(y_train))

print("Count of y datas:", len(y_valid))

xg = XGBClassifier(objective="multi:softmax", eval_metric="merror", random_state=4242, n_jobs=-1, n_estimators=30, predictor='gpu_predictor')
# xg.fit()

# ctf = OneVsRestClassifier(xg, n_jobs=-1)
# print(ctf)
print("Fit model...")
xg.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_valid, y_valid)])


# ctf.fit(x_train, y_train)
print("Predict data...")
pred = xg.predict_proba(x_valid)

print("Process data...")
# print(pred)
# print(pred.shape)

# <glc_id;species_glc_id;probability;rank>

result = pd.DataFrame(columns=['glc_id', 'species_glc_id', 'probability', 'rank', 'real_species_glc_id'])

for i in range(0, len(pred)):
    current_pred = pred[i]
    current_glc_id = i
    # pred_ranked = pd.Series(current_pred).rank(ascending=False, method="min")
    pred_r = rankdata(current_pred, method="ordinal")
    # absteigend sortieren
    pred_r = len(train_species_ids) - pred_r + 1
    glc_id_array = [int(current_glc_id)] * len(train_species_ids)
    # glc_id_array = pd.to_numeric(glc_id_array, downcast='integer')

    sol_array = [int(list(y_valid)[i])] * len(train_species_ids)

    percentile_list = pd.DataFrame({
        'glc_id': glc_id_array,
        'species_glc_id': train_species_ids,
        'probability': current_pred,
        'rank': pred_r,
        'real_species_glc_id': sol_array,
    })

    # macht aus int float!
    # percentile_list = pd.DataFrame(np.column_stack([glc_id_array, train_species_ids, current_pred, pred_r]), columns=['glc_id', 'species_glc_id', 'probability', 'rank'])
    result = pd.concat([result, percentile_list], ignore_index=True)
result = result.reindex(columns=('glc_id', 'species_glc_id', 'probability', 'rank', 'real_species_glc_id'))

result = result[result.species_glc_id == result.real_species_glc_id]

# print(result)

print("Calculate MRR-Score...")
sum = 0.0
Q = len(result.index)

print("Dropped valpredictions because species was only in valset:", len(y_valid) - Q)

# MRR berechnen
for index, row in result.iterrows():
    sum += 1 / float(row["rank"])

mrr_score = 1.0 / Q * sum
print("MRR-Score:", mrr_score)

result.drop(['real_species_glc_id'], axis=1, inplace=True)
result.to_csv(data_paths.submission_val, index=False, sep=";")
print("Total duration (s):", time.time() - all_start)