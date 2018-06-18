def mrr_score(ranks):
    '''Calculates the mrr for a list of ranks. Remarks: Ranks can also be zero.'''
    Q = len(ranks)
    sum = 0.0

    for rank in ranks:
        if rank != 0:
            sum += 1 / float(rank)

    mrr_score = 1.0 / Q * sum

    return mrr_score

def mrr_score_df(df):
    '''Calculates the mrr for a DataFrame where the rank's colum is named 'rank'.'''
    assert "rank" in df.columns.values
    return mrr_score(list(df["rank"].values))
