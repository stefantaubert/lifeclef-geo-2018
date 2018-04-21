import module_support_main
import data_paths_main as data_paths
import pandas as pd
import mrr
from tqdm import tqdm
import numpy as np
import settings_main as settings
from sklearn.model_selection import train_test_split
from itertools import chain

def evaluate_xgb_groups():
    print("Evaluate submission...")
    print("Load data...")
    df = pd.read_csv(data_paths.xgb_submission) 
    x_text = pd.read_csv(data_paths.train)
    y = x_text["species_glc_id"]
    named_groups = np.load(data_paths.named_groups)
    
    c_classes = 0
    for _, species in named_groups.items():
        for _ in species:
            c_classes += 1

    _, _, _, y_valid = train_test_split(x_text, y, test_size=settings.train_val_split, random_state=settings.seed)
    
    print("Calculate MRR-Score...")    
    ranks = get_ranks(df, y_valid, c_classes)
    mrr_score = mrr.mrr_score(ranks)
    print("MRR-Score:", mrr_score * 100,"%")

def evaluate_xgb_normal():
    print("Evaluate submission...")
    print("Load data...")
    df = pd.read_csv(data_paths.xgb_submission)
    x_text = pd.read_csv(data_paths.train)
    y = x_text["species_glc_id"]
    c_classes = len(np.load(data_paths.xgb_species_map))

    _, _, _, y_valid = train_test_split(x_text, y, test_size=settings.train_val_split, random_state=settings.seed)
    
    print("Calculate MRR-Score...")    
    ranks = get_ranks(df, y_valid, c_classes)
    mrr_score = mrr.mrr_score(ranks)
    print("MRR-Score:", mrr_score * 100,"%")

def evaluate_results_from_files(submission_path, gt_path, species_map_path):
    print("Evaluate submission...")
    print("Load data...")
    df = pd.read_csv(submission_path)
    y = np.load(gt_path)
    c_classes = len(np.load(species_map_path))
    
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
    evaluate_xgb_normal()
    #evaluate_current_training_results()