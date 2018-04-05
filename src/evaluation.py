from settings import *
import numpy as np
import data_paths
import pickle
from scipy.stats import rankdata
import pandas as pd
from sklearn.model_selection import train_test_split

def evaluate_with_mrr():
    print("Evaluate submission...")
    
    df = pd.read_csv(data_paths.submission_val)
 
    print("Calculate MRR-Score...")
    sum = 0.0
    Q = len(df.index)

    # MRR berechnen
    for index,row in df.iterrows():
        sum += 1 / float(row["rank"])

    mrr_score = 1.0 / Q * sum

    print("MRR-Score:", mrr_score * 100,"%")

if __name__ == '__main__':
    evaluate_with_mrr()
