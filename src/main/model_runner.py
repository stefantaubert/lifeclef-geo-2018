import module_support_main
import time 
import XGBoostModel
import XGBoostModelGroups
import submission
import evaluation

def startXGBoost():
    ## preprocessing is already done
    start_time = time.time()

    XGBoostModel.XGBModel().run()
    submission.make_xgb_submission()
    evaluation.evaluate_with_mrr()

    print("Total duration:", time.time() - start_time, "s")

def startXGBoostGroups():
    ## preprocessing is already done
    start_time = time.time()

    XGBoostModelGroups.XGBModel().run()
    submission.make_xgb_groups_submission()
    evaluation.evaluate_with_mrr()

    seconds = time.time() - start_time
    print("Total duration:", round(seconds / 60, 2), "min")

if __name__ == "__main__":
    startXGBoostGroups()