import pandas as pd

def mrr_score(ranks):
    Q = len(ranks)
    sum = 0.0

    # MRR berechnen
    for rank in ranks:
        sum += 1 / float(rank)

    mrr_score = 1.0 / Q * sum

    return mrr_score

def mrr_score_df(df):
    assert "rank" in df.columns.values
    return mrr_score(list(df["rank"].values))
