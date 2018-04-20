import module_support

import main_preprocessing
import model_runner

main_preprocessing.extract_groups()
model_runner.startXGBoostGroups()