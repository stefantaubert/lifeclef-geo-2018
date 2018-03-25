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

csv = pd.read_csv(data_paths.occurrences_train, sep=';', nrows=5000)
csv["glc_id"] = csv["patch_id"]

#csv = csv.fillna('0')
# print(len(csv.index))
#
# csv = csv.dropna(axis=0, how='any')
# print(len(csv.index))
# def select_rows(df,search_strings):
#     unq,IDs = np.unique(df,return_inverse=True)
#     unqIDs = np.searchsorted(unq,search_strings)
#     return df[((IDs.reshape(df.shape) == unqIDs[:,None,None]).any(-1)).all(0)]
#
# rows_without_NA = csv[csv.isin([nan])]
# print(len(rows_without_NA))
#
# #rows_with_NA = csv[csv == 'NA'].dropna(how='all')
#
# for index, row in csv.iterrows():
#     for val in row.values:
#         if str(val) == "1024178204":
#             print(row)
#
#
# rows_with_NA = select_rows(csv, ['NA'])
# print(rows_with_NA)
#
# csv = csv.drop(rows_with_NA)
#
# print(len(csv.index))
#
#
# # macht keinen sinn
# csv = csv.replace(['NA'], ['0'])
#print(csv.head())
x_train = csv[[
        'species_glc_id', 'glc_id',
        # 'patch_id',
        'chbio_1','chbio_2','chbio_3', 'chbio_4', 'chbio_5', 'chbio_6',
        'chbio_7', 'chbio_8', 'chbio_9', 'chbio_10', 'chbio_11', 'chbio_12',
        'chbio_13', 'chbio_14', 'chbio_15', 'chbio_16', 'chbio_17', 'chbio_18',
        'chbio_19', 'alti',
        'awc_top', 'bs_top', 'cec_top', 'crusting', 'dgh', 'dimp','erodi', 'oc_top', 'pd_top', 'text',
        'proxi_eau_fast','clc'
     ]]
old_rowcount = len(x_train.index)
x_train = x_train.dropna(axis=0, how='any')
print("Count of dropped rows with 'nan'", old_rowcount - len(x_train.index), "of", old_rowcount)


#x_train.to_csv(data_paths.train_features, index=False)
#x_train = pd.read_csv(data_paths.features_train)
species = set(csv["species_glc_id"].values)
# print("All species:", species)
print("All species count:", len(species))

# Ausgabedaten erstellen.
y_train = x_train.species_glc_id
#y_train = array2
x_train.drop(["species_glc_id"],axis=1, inplace=True)

# Trainings-Set und Validierungs-Set erstellen. Das Validierungs-Set enthält 10% aller Trainings-Daten.
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=4242)
print("Validationset rows:", len(x_valid.index))
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
print("Unknown species in trainset:", len(species) - len(train_species_ids))

x_valid_ids = list(x_valid.glc_id)
x_valid.drop(["glc_id"],axis=1, inplace=True)

# print("Validationfeatures:", x_valid)
# print("Validation glc_ids:", x_valid_ids)
# print("Count Validation glc_ids:", len(x_valid_ids))
print("Species only in validationset:", len(set(val_species_ids).difference(set(train_species_ids))) )


if (False):
    clf = DecisionTreeClassifier()
    clf.n_classes_ = len(species)
    clf.fit(x_train, y_train)
    pred = clf.predict_proba(x_valid)
else:
    ctf = OneVsRestClassifier(XGBClassifier(n_jobs=-1, seed=4242))
    print("Fit model...")
    ctf.fit(x_train, y_train)
    print("Predict data...")
    pred = ctf.predict_proba(x_valid)

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

result_clean = pd.DataFrame(columns=['glc_id', 'species_glc_id', 'probability', 'rank', 'real_probability'])

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

result.to_csv(data_paths.submission_val, index=False, sep=";")
print("Total duration (s):", time.time() - all_start)