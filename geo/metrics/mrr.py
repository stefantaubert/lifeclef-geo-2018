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
    '''Calculates the mrr for a DataFrame where the rank's column is named 'rank'.'''
    assert "rank" in df.columns.values
    return mrr_score(list(df["rank"].values))

# class mrr_eval():
#     '''Calculate mrr for XGB-Training'''
#     def __init__(self, classes, y_valid):
#         self.classes = classes
#         self.y_valid = y_valid
#         self.class_count = len(self.classes)

#     def evalute(self, y_predicted, y_true):
#         print("evaluate")
#         glc = [x for x in range(len(y_predicted))]
#         subm = submission_maker._make_submission(self.class_count, self.classes, y_predicted, glc)
#         ranks = get_ranks.get_ranks(subm, self.y_valid, self.class_count)
#         mrr_score = mrr.mrr_score(ranks)
#         return ("mrr", mrr_score)
