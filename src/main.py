import module_support
import main_preprocessing
import model_runner

def run_xgb_groups():
    main_preprocessing.extract_groups()
    model_runner.startXGBoostGroups()

if __name__ == "__main__":
    run_xgb_groups()