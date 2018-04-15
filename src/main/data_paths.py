import settings

root_lines = open("../config/data_root").read().split('\n')
root = root_lines[0]

patch_train = root + "patchTrain"
patch_test = root + "patchTest"

occurrences_train = root + "occurrences_train.csv"
occurrences_test = root + "occurrences_test.csv"
occurrences_train_gen = root + "occurrences_train_gen_" + str(settings.pixel_count) + ".csv"
occurrences_test_gen = root + "occurrences_test_gen_" + str(settings.pixel_count) + ".csv"

submission_val = root + "submissions/submission_val.csv"

train_samples = root + "ImageLists/Train/samples.npy"
train_samples_species_map = root + "ImageLists/Train/species_map.py"

test_samples = root + 'ImageLists/Test/samples.npy'

current_training_gt = root + "Current_Training_Results/gt.npy"
current_training_results = root + "Current_Training_Results/results.npy"
current_training_species_map = root + "Current_Training_Results/species_map.py"
current_training_submission = root + "Current_Training_Results/submission.csv"

current_training_model = root + 'Current_Training_Results/model.h5'