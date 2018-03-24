import time
import pandas as pd
import numpy as np
import data_paths
import time
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import rankdata
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score, recall_score, precision_score, f1_score
from xgboost import XGBClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
import xgboost as xgb
import rank_metrics
import data_paths

all_start = time.time()

csv = pd.read_csv(data_paths.occurrences_train, sep=';', nrows=500)
csv["glc_id"] = csv["patch_id"]

csv = csv.fillna('0')
x_train = csv[['glc_id',
   # 'patch_id',
    'chbio_1','chbio_2','chbio_3', 'chbio_4', 'chbio_5', 'chbio_6',
     # 'chbio_7', 'chbio_8', 'chbio_9', 'chbio_10', 'chbio_11', 'chbio_12', 'chbio_13', 'chbio_14', 'chbio_15', 'chbio_16', 'chbio_17', 'chbio_18', 'chbio_19', 'alti', 'awc_top', 'bs_top', 'cec_top', 'crusting', 'dgh', 'dimp', 'erodi', 'oc_top', 'pd_top', 'text', 'proxi_eau_fast','clc'
                      ]]

#x_train.to_csv(data_paths.train_features, index=False)
#x_train = pd.read_csv(data_paths.features_train)
species = set(csv["species_glc_id"].values)
print("All species:", species)
print("All species count:", len(species))

array  = [0 for i in range(0, len(set(csv["species_glc_id"].values)))]
array2 = [array for i in range(0, len(csv.index))]
print(array)
print(array2)
# Ausgabedaten erstellen.
y_train = csv.species_glc_id
#y_train = array2

# Trainings-Set und Validierungs-Set erstellen. Das Validierungs-Set enthält 10% aller Trainings-Daten.
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=4242)
print("Validationset rows:", len(x_valid.index))
train_species_ids = list(set(y_train))
val_species_ids = list(set(y_valid))

print("Trainspecies:", train_species_ids)
print("Trainspecies count:", len(train_species_ids))
print("Trainspecies last:", list(train_species_ids)[-1])

print("Valspecies:", val_species_ids)
print("Valspecies count:", len(val_species_ids))
print("Valspecies last:", list(val_species_ids)[-1])

#x_train.drop(['patch_id'], axis=1)

x_train_ids = x_train.glc_id
x_train.drop(["glc_id"],axis=1, inplace=True)

print("Trainfeatures:", x_train)
print("Train glc_ids:", x_train_ids)
print("Unknown Spezies:", len(species) - len(train_species_ids))

x_valid_ids = list(x_valid.glc_id)
x_valid.drop(["glc_id"],axis=1, inplace=True)

print("Validationfeatures:", x_valid)
print("Validation glc_ids:", x_valid_ids)
print("Count Validation glc_ids:", len(x_valid_ids))
print("Species only in validationset:", len(set(val_species_ids).difference(set(train_species_ids))) )
print(type(x_valid_ids))

if (False):
    clf = DecisionTreeClassifier()
    clf.n_classes_ = len(species)
    clf.fit(x_train, y_train)
    pred = clf.predict_proba(x_valid)
else:
    ctf = OneVsRestClassifier(XGBClassifier(n_jobs=-1))
    ctf.fit(x_train, y_train)
    pred = ctf.predict_proba(x_valid)

print(pred)
print(pred.shape)

#<glc_id;species_glc_id;probability;rank>

result = pd.DataFrame(columns=['glc_id', 'species_glc_id', 'probability', 'rank'])

for i in range(0, len(pred)):
    current_pred = pred[i]
    current_glc_id = int(x_valid_ids[i])
    #pred_ranked = pd.Series(current_pred).rank(ascending=False, method="min")
    pred_r = rankdata(current_pred, method="ordinal")
    # absteigend sortieren
    pred_r = len(train_species_ids) - pred_r + 1
    glc_id_array = [int(current_glc_id)] * len(train_species_ids)
    #glc_id_array = pd.to_numeric(glc_id_array, downcast='integer')

    percentile_list = pd.DataFrame({
        'glc_id': glc_id_array,
        'species_glc_id': train_species_ids,
        'probability': current_pred,
        'rank': pred_r
    })

    # macht aus int float!
    #percentile_list = pd.DataFrame(np.column_stack([glc_id_array, train_species_ids, current_pred, pred_r]), columns=['glc_id', 'species_glc_id', 'probability', 'rank'])
    result = pd.concat([result, percentile_list], ignore_index=True)
result = result.reindex(columns=('glc_id', 'species_glc_id', 'probability', 'rank'))
print(result)
result.to_csv(data_paths.submission_val, index=False, sep=";")
