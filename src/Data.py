import pandas as pd
import numpy as np
import data_paths

class Data:
    def __init__(self):
        pass

    def load_train(self):
        self.train = pd.read_csv(data_paths.occurrences_train_gen)
        self.species = sorted(self.train.species_glc_id.unique())
        self.species_count = len(self.species)
        r = [i + 1 for i in range(len(self.species))]
        assert self.species == r
    
    def load_test(self):
        self.test = pd.read_csv(data_paths.occurrences_test_gen)
