import module_support
import main_preprocessing
import time, datetime
import XGBoostModelNative
import XGBoostModelGroups
import submission
import evaluation
import LogisticRegressionModel
import VectorModel
import RandomModel
import Log
import data_paths_global as data_paths
import settings_main as settings

def writeLog(title, start, end, duration, model, mrr):
    log_text = str("{}\n--------------------\nMRR-Score: {}\nStarted: {}\nFinished: {}\nDuration: {}min\nSuffix: {}\nTraincolumns: {}\nSeed: {}\nSplit: {}\n".format
    (
        title,
        str(mrr), 
        str(start), 
        str(end),
        str(duration),
        data_paths.get_suffix_prot(),
        ", ".join(model.train_columns),
        settings.seed,
        settings.train_val_split,
    ))
    log_text += "Modelparams:\n"
    params = ["- {}: {}\n".format(x, y) for x, y in model.params.items()]
    log_text += "".join(params) + "============================="
    Log.write(log_text)
    print(log_text)

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
    #startRandomModel()
    #startXGBRegression()
    #startXGBRegressionGroups()
    #startXGBoostNative()
    #predictTestDataXGBNative(36)
    #startXGBoost(False)
    #startXGBoostGroups()
