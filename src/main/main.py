import time 
import XGBoostModel
import XGBoostModelGroups
import submission_maker
import evaluation

def startXGBoost():
    ## preprocessing is already done
    start_time = time.time()

    XGBoostModel.XGBModel().run()
    submission_maker.make_xgb_submission()
    evaluation.evaluate_with_mrr()

    print("Total duration:", time.time() - start_time, "s")

def startXGBoostGroups():
    ## preprocessing is already done
    start_time = time.time()

    XGBoostModelGroups.XGBModel().run()
    submission_maker.make_xgb_groups_submission()
    evaluation.evaluate_with_mrr()

    print("Total duration:", time.time() - start_time, "s")

if __name__ == "__main__":
    startXGBoost()