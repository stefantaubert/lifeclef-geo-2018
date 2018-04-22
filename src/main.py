import module_support
import main_preprocessing
import time 
import XGBoostModel
import XGBoostModelGroups
import submission
import evaluation

def startXGBoost():
    start_time = time.time()

    main_preprocessing.create_datasets()
    XGBoostModel.XGBModel().run(True)
    submission.make_xgb_submission()
    submission.make_xgb_test_submission()
    evaluation.evaluate_xgb()

    seconds = time.time() - start_time
    print("Total duration:", round(seconds / 60, 2), "min")

def startXGBoostGroups():
    ## preprocessing is already done
    start_time = time.time()

    main_preprocessing.create_datasets()
    XGBoostModelGroups.XGBModel().run(True)
    submission.make_xgb_groups_submission()
    submission.make_xgb_groups_test_submission()
    evaluation.evaluate_xgb()

    seconds = time.time() - start_time
    print("Total duration:", round(seconds / 60, 2), "min")

if __name__ == "__main__":
    startXGBoostGroups()