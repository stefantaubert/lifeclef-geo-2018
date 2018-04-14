import data_paths
import pandas as pd
import mrr
from tqdm import tqdm
import numpy as np
import settings 
from sklearn.model_selection import train_test_split
from itertools import chain

def evaluate_with_mrr():
    print("Evaluate submission...")
    print("Load data...")
    df = pd.read_csv(data_paths.submission_val)
    x_text = pd.read_csv(data_paths.occurrences_train_gen)
    y = np.load(data_paths.y_ids)
    c_classes = len(np.load(data_paths.species_map_training))
    
    x_train, x_valid, y_train, y_valid = train_test_split(x_text, y, test_size=settings.train_val_split, random_state=settings.seed)
    
    print("Calculate MRR-Score...")    
    ranks = get_ranks(df, y_valid, c_classes)
    mrr_score = mrr.mrr_score(ranks)
    print("MRR-Score:", mrr_score * 100,"%")


def evaluate_current_training_results():
    print("Evaluate submission...")
    print("Load data...")
    df = pd.read_csv(data_paths.current_training_submission)
    y = np.load(data_paths.current_training_gt)
    c_classes = len(np.load(data_paths.current_training_species_map))
    
    print("Calculate MRR-Score...")    
    ranks = get_ranks(df, y, c_classes)
    mrr_score = mrr.mrr_score(ranks)
    print("MRR-Score:", mrr_score * 100,"%")

def get_ranks(submissions_df, solutions, c_classes):
    # Erzeuge ein Array das die Lösungsspecies pro durchlauf enthält zb [3, 3, 3, 4, 4, 4] bei 3 Klassen und 2 Predictions.
    sol_array = [[s] * c_classes for s in solutions]
    sol_array = list(chain.from_iterable(sol_array))

    assert len(sol_array) == len(submissions_df.index)

    submissions_df["sol_glc_id"] = sol_array
    submissions_df = submissions_df[submissions_df["species_glc_id"] == submissions_df["sol_glc_id"]]
   
    return submissions_df["rank"].values



if __name__ == '__main__':
    #evaluate_with_mrr()
    evaluate_current_training_results()