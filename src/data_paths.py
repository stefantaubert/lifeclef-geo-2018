root_lines = open("../config/data_root").read().split('\n')
root = root_lines[0]

occurrences_train = root + "occurrences_train.csv"
features_train = root + "train_features.csv"
submission_val = root + "submission_val.csv"