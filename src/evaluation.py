import data_paths
import pandas as pd
import mrr
from tqdm import tqdm
import numpy as np
import settings 
from sklearn.model_selection import train_test_split

def evaluate_with_mrr():
    print("Evaluate submission...")
    df = pd.read_csv(data_paths.submission_val)
    x_text = pd.read_csv(data_paths.occurrences_train_gen)
    y = np.load(data_paths.y_ids)
    x_train, x_valid, y_train, y_valid = train_test_split(x_text, y, test_size=settings.train_val_split, random_state=settings.seed)
    
    print("Calculate MRR-Score...")    
    ranks = get_ranks(df, y_valid)
    mrr_score = mrr.mrr_score(ranks)
    print("MRR-Score:", mrr_score * 100,"%")

def get_ranks(submissions_df, solutions):
    final_ranks = []
    current_solution_index = 0
    for i, current_submission in tqdm(submissions_df.iterrows()):
        if current_submission["species_glc_id"] == solutions[current_solution_index]:
            final_ranks.append(current_submission["rank"])
            if current_solution_index < len(solutions) - 1:
                current_solution_index += 1

    return final_ranks

if __name__ == '__main__':
    evaluate_with_mrr()
