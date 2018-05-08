import module_support_main
import pandas as pd
import numpy as np
import time
from scipy.stats import rankdata
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import data_paths_main as data_paths
import submission_maker
import evaluation
import settings_main as settings
import time
import random
from tqdm import tqdm

random.seed = settings.seed

class Model():
    def __init__(self):
        x_text = pd.read_csv(data_paths.train)
        x_test = pd.read_csv(data_paths.test)
        
        y = x_text["species_glc_id"]
       
        # species_count = np.load(data_paths.y_array).shape[1]
        self.class_names = list(np.unique(y))
        #np.save(data_paths.xgb_species_map, classes_)

        self.x_train, self.x_valid, self.y_train, self.y_valid = train_test_split(x_text, y, test_size=settings.train_val_split, random_state=settings.seed)

        np.save(data_paths.xgb_species_map, self.class_names)
        np.save(data_paths.xgb_glc_ids, self.x_valid["patch_id"])

        self.x_train = self.x_train
        self.x_valid = self.x_valid
        self.x_test = x_test

    def run(self):
        print("Run model...")
        self.species_count = len(self.class_names)
        self.fake_propabilities = [(self.species_count - i) / self.species_count for i in range(self.species_count)]
        
        test_predictions = []
        for i in tqdm(range(len(self.x_test.index))):
            random_species = random.sample(self.class_names, self.species_count)
            probs = list(self.fake_propabilities)
            _, sorted_probs = zip(*sorted(zip(random_species, probs)))
            test_predictions.append(sorted_probs)

        print("Finished.")
        print("Saving results...")
        np.save(data_paths.random_test_prediction, np.array(test_predictions))
        np.save(data_paths.random_species, np.array(self.class_names))
        print("Saving completed", data_paths.random_species, data_paths.random_test_prediction)

if __name__ == '__main__':
    m = Model()
    m.run()


# def binary_search(a, x, lo=0, hi=None):  # can't use a to specify default for hi
#     hi = hi if hi is not None else len(a)  # hi defaults to len(a)   
#     pos = bisect_left(a, x, lo, hi)  # find insertion position
#     return (pos if pos != hi and a[pos] == x else -1)  # don't walk off the end

# def run_Model():
#     print("Run model...")
#     #x_text = np.load(data_paths.x_text)

#     x_text = pd.read_csv(data_paths.occurrences_train_gen, sep=";")
#     species_ids = x_text["species_glc_id"].values
#     counter = Counter(species_ids)
#     count_unique_species = len(set(species_ids))
#     top_species = counter.most_common()[:count_most_common]
#     y = np.load(data_paths.y_ids)

#     species, occ = zip(*top_species)
#     #print(species)

#     species = list(species)
#     species.sort()
#     print(species)
#     x_train, x_valid, y_train, y_valid = train_test_split(x_text, y, test_size=settings.train_val_split, random_state=settings.seed)

#     prediction = []
#     print(len(y_valid))
#     for i in tqdm(range(len(y_valid))):
#         current_prediction = []
#         for j in range(1, count_unique_species+1):
#             ind  = binary_search(species, j)

#             if ind == -1:
#                 predicted_prob = 0
#             else:
#                 predicted_prob = random.uniform(0, 1)
#             current_prediction.append(predicted_prob)
#         prediction.append(current_prediction)

#     np.save(data_paths.prediction, prediction)


# if __name__ == '__main__':
#     start_time = time.time()

#     # DataReader.read_and_write_data()
#     run_Model()
#     submission_maker.make_submission()
#     evaluation.evaluate_with_mrr()

#     print("Total duration:", time.time() - start_time, "s")