import time
import pandas as pd
import numpy as np
import data_paths
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score, recall_score, precision_score, f1_score
from xgboost import XGBClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
import xgboost as xgb
import rank_metrics
import data_paths

all_start = time.time()

csv = pd.read_csv(data_paths.occurrences_train, error_bad_lines=False, sep=';', low_memory=False)

train_features = csv[['chbio_1',
     'chbio_2','chbio_3', 'chbio_4', 'chbio_5', 'chbio_6', 'chbio_7', 'chbio_8', 'chbio_9', 'chbio_10', 'chbio_11', 'chbio_12', 'chbio_13', 'chbio_14', 'chbio_15', 'chbio_16', 'chbio_17', 'chbio_18', 'chbio_19', 'alti', 'awc_top', 'bs_top', 'cec_top', 'crusting', 'dgh', 'dimp', 'erodi', 'oc_top', 'pd_top', 'text', 'proxi_eau_fast','clc'
                      ]]

#train_features.to_csv(data_paths.train_features, index=False)
#x_train = pd.read_csv(data_paths.train_features)
df_train = pd.read_csv(data_paths.occurrences_train, error_bad_lines=False, sep=';', low_memory=False)
x_train = train_features

# Ausgabedaten erstellen.
y_train = df_train.species_glc_id

# Trainings-Set und Validierungs-Set erstellen. Das Validierungs-Set enthält 10% aller Trainings-Daten.
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1, random_state=4242)

ctf = OneVsRestClassifier(XGBClassifier(n_jobs=-1))
ctf.fit(x_train, y_train)
print(ctf)

y_pred = ctf.predict(x_valid)
print(y_pred)

print(accuracy_score(y_valid, y_pred))