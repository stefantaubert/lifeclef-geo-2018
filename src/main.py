import module_support
import main_preprocessing
import time 
import XGBoostModel
import XGBoostModelGroups
import submission
import evaluation

def startXGBoostNavive(with_test):
    start_time = time.time()

    main_preprocessing.create_datasets()
    XGBoostModel.XGBModel().run(with_test)
    submission.make_xgb_submission()
    evaluation.evaluate_xgb()
    if with_test:
        submission.make_xgb_test_submission()

    seconds = time.time() - start_time
    print("Total duration:", round(seconds / 60, 2), "min")

def startXGBoost(with_test):
    start_time = time.time()

    main_preprocessing.create_datasets()
    XGBoostModel.XGBModel().run(with_test)
    submission.make_xgb_submission()
    evaluation.evaluate_xgb()
    if with_test:
        submission.make_xgb_test_submission()

    seconds = time.time() - start_time
    print("Total duration:", round(seconds / 60, 2), "min")

def startXGBoostGroups():
    ## preprocessing is already done
    start_time = time.time()

    main_preprocessing.create_datasets()
    main_preprocessing.extract_groups()
    XGBoostModelGroups.XGBModel().run(True)
    submission.make_xgb_groups_submission()
    evaluation.evaluate_xgb()
    submission.make_xgb_groups_test_submission()

    seconds = time.time() - start_time
    print("Total duration:", round(seconds / 60, 2), "min")

if __name__ == "__main__":
    startXGBoost(False)
    #startXGBoostGroups()