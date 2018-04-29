import module_support_main
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import data_paths_main as data_paths
import settings_main as settings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from scipy.sparse import hstack
from joblib import Parallel, delayed
import multiprocessing
import submission_maker
import get_ranks
import mrr
import main_preprocessing
main_preprocessing.create_datasets()    
# what are your inputs, and what operation do you want to 
# perform on each input. For example...
inputs = range(10) 
def processInput(i):
	return i * i

print("Run model...")
#x_text = np.load(data_paths.x_text)
x_text = pd.read_csv(data_paths.train)
y = x_text["species_glc_id"]
train_columns = ['bs_top', 'alti', 'chbio_12', 'chbio_15', 'chbio_17', 'chbio_3', 'chbio_6', 'clc', 'crusting', 'dimp']
# species_count = np.load(data_paths.y_array).shape[1]
class_names = np.unique(y)
#np.save(data_paths.xgb_species_map, classes_)

x_train, x_valid, y_train, y_valid = train_test_split(x_text, y, test_size=settings.train_val_split, random_state=settings.seed)

np.save(data_paths.xgb_species_map, class_names)
np.save(data_paths.xgb_glc_ids, x_valid["patch_id"])

x_train = x_train[train_columns]
x_valid = x_valid[train_columns]

scores = []
submission = {}

def calc_class(class_name):
    train_target = list(map(lambda x: 1 if x == class_name else 0, y_train))
    val_target = list(map(lambda x: 1 if x == class_name else 0, y_valid))
    #print(train_target)
    classifier = LogisticRegression(C=0.1, solver='sag', n_jobs=-1, random_state=settings.seed, max_iter=1)

    #cv_score = np.mean(cross_val_score(classifier, x_train, train_target, cv=3, scoring='roc_auc'))
    #scores.append(cv_score)
    #print('CV score for class {} is {}'.format(class_name, cv_score))
    classifier.fit(x_train, train_target)
    pred = classifier.predict_proba(x_valid)
    #print(pred)
    pred_real = pred[:, 1] # second is for class is 1
    #print("acc", accuracy_score(val_target, pred_real.round()))
    score = roc_auc_score(val_target, pred_real)
    print('ROC AUC score for class {} is {}'.format(class_name, score.round()))
    scores.append(score)
    submission[class_name] = pred_real

def evalute(y_predicted, y_true, classes):
        print("evaluate")
        glc = [x for x in range(len(y_predicted))]
        subm = submission_maker._make_submission(len(classes), classes, y_predicted, glc)
        ranks = get_ranks.get_ranks(subm, y_true, len(classes))
        mrr_score = mrr.mrr_score(ranks)
        return ("mrr", mrr_score)

if __name__ == '__main__':
    
    num_cores = multiprocessing.cpu_count()
    Parallel(n_jobs=num_cores)(delayed(calc_class)(class_name) for class_name in tqdm(class_names))

    result = submission.values()
    assert len(result) == len(class_names)
    assert len(result[0]) == len(x_valid.index)
    arr = np.array(result)
    result = arr.T
    print(result)
    evalute(result, y_valid, class_names)
    print('Total ACC score is {}'.format(np.mean(scores)))