import data_paths
import pandas as pd
import mrr

def evaluate_with_mrr():
    print("Evaluate submission...")
    df = pd.read_csv(data_paths.submission_val)
    print("Calculate MRR-Score...")
    mrr_score = mrr.mrr_score_df(df)
    print("MRR-Score:", mrr_score * 100,"%")

if __name__ == '__main__':
    evaluate_with_mrr()
