import module_support
import main_preprocessing
import time, datetime
import XGBoostModelNative
import XGBoostModelGroups
import submission
import evaluation
import LogisticRegressionModel
import Log
import data_paths_global as data_paths

def startXGBRegression():
    start_time = time.time()
    start_datetime = datetime.datetime.now().time()
    print("Start:", start_datetime)
    main_preprocessing.create_datasets()
    m = LogisticRegressionModel.Model()
    m.run()
    submission.make_logistic_test_submission()
    mrr = evaluation.evaluate_xgb_regression()
    end_date_time = datetime.datetime.now().time()
    print("End:", end_date_time)
    seconds = time.time() - start_time
    duration_min = round(seconds / 60, 2)
    print("Total duration:", duration_min, "min")
    Log.write("XGBoost Regression Model\nMRR-Score: {}\nStarted: {}\nFinished: {}\nDuration: {}min\nSuffix: {}\nTraincolumns: {}\n==========".format
    (
        str(mrr), 
        str(start_datetime), 
        str(end_date_time),
        str(duration_min),
        data_paths.get_suffix_pro(),
        ", ".join(m.train_columns))
    )

def startXGBoostNative():
    start_time = time.time()
    print("Start:", datetime.datetime.now().time())
    main_preprocessing.create_datasets()
    XGBoostModelNative.XGBModelNative().run()
    
    # submission.make_xgb_submission()
    # evaluation.evaluate_xgb()
    
    print("End:", datetime.datetime.now().time())
    seconds = time.time() - start_time
    print("Total duration:", round(seconds / 60, 2), "min")

def predictTestDataXGBNative(iteration):
    start_time = time.time()
    print("Start:", datetime.datetime.now().time())
    main_preprocessing.create_datasets()
    XGBoostModelNative.XGBModelNative().predict_test_set_from_saved_model(iteration)
    submission.make_xgb_test_submission()    
    print("End:", datetime.datetime.now().time())
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
    startXGBRegression()
    #startXGBoostNative()
    #predictTestDataXGBNative(0)
    #startXGBoost(False)
    #startXGBoostGroups()
