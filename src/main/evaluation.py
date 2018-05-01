import module_support_main
import data_paths_main as data_paths
import pandas as pd
import mrr
import numpy as np
import settings_main as settings
from sklearn.model_selection import train_test_split
import get_ranks
import submission_maker

def evaluate_xgb():
    print("Evaluate submission...")
    print("Load data...")
    df = pd.read_csv(data_paths.xgb_submission)
    x_text = pd.read_csv(data_paths.train)
    y = x_text["species_glc_id"]

    _, _, _, y_valid = train_test_split(x_text, y, test_size=settings.train_val_split, random_state=settings.seed)
    
    print("Calculate MRR-Score...")    
    ranks = get_ranks.get_ranks_df(df, y_valid, settings.TOP_N_SUBMISSION_RANKS)
    mrr_score = mrr.mrr_score(ranks)
    print("MRR-Score:", mrr_score * 100,"%")

def evaluate_xgb_regression():
    print("Evaluate submission...")
    print("Load data...")
    species = np.load(data_paths.regression_species)
    predictions = np.load(data_paths.regression_prediction)
    x_text = pd.read_csv(data_paths.train)
    y = x_text["species_glc_id"]
    _, _, _, y_valid = train_test_split(x_text, y, test_size=settings.train_val_split, random_state=settings.seed)

    assert len(predictions) == len(y_valid.index)
    assert len(predictions[0]) == len(species)
    
    glc = [x for x in range(len(predictions))]
    subm = submission_maker._make_submission(len(species), species, predictions, glc)
    ranks = get_ranks.get_ranks(subm, y_valid, len(species))
    mrr_score = mrr.mrr_score(ranks)
    print("MRR-Score:", mrr_score * 100,"%")
    return mrr_score

def evaluate_results_from_files(submission_path, gt_path, species_map_path):
    print("Evaluate submission...")
    print("Load data...")
    df = pd.read_csv(submission_path)
    y = np.load(gt_path)
    
    print("Calculate MRR-Score...")
    ranks = get_ranks.get_ranks_df(df, y, settings.TOP_N_SUBMISSION_RANKS)
    mrr_score = mrr.mrr_score(ranks)
    print("MRR-Score:", mrr_score * 100,"%")

if __name__ == "__main__":
    evaluate_xgb()