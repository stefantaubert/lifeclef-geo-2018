root_lines = open("../config/data_root").read().split('\n')
root = root_lines[0]

occurrences_train = root + "occurrences_train.csv"
train_features = root + "train_features.csv"
