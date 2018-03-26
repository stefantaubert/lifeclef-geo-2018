from xgboost import XGBClassifier

class XGBCl():

    def fit(self, X, y, sample_weight=None, eval_set=None, eval_metric=None,
            early_stopping_rounds=None, verbose=True, xgb_model=None):
        XGBClassifier().fit(X, y)
