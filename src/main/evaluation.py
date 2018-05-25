import module_support_main
import pandas as pd
import mrr
import numpy as np
import settings_main as settings
import get_ranks

def evaluate_results_from_files(submission_path, gt_path, species_map_path):
    print("Evaluate submission...")
    print("Load data...")
    df = pd.read_csv(submission_path)
    y = np.load(gt_path)
    
    print("Calculate MRR-Score...")
    ranks = get_ranks.get_ranks_df(df, y, settings.TOP_N_SUBMISSION_RANKS)
    mrr_score = mrr.mrr_score(ranks)
    print("MRR-Score:", mrr_score * 100,"%")
