import numpy as np
import pandas as pd
import data_paths
from tqdm import tqdm
import tifffile
import pickle
import settings
import sys

class Data:
    '''Loads test and trainset and rounds the values.'''
    
    def load_train(self):
        self.train = pd.read_csv(data_paths.occurrences_train_gen)
        self.train = self.train.round(settings.round_data_ndigits)
        self.species = sorted(self.train.species_glc_id.unique())
        self.species_count = len(self.species)

        r = [i + 1 for i in range(len(self.species))]
        assert self.species == r
    
    def load_test(self):
        self.test = pd.read_csv(data_paths.occurrences_test_gen)
        self.test = self.test.round(settings.round_data_ndigits)

    def load_datasets(self):
        print("Loading trainset...")
        self.load_train()

        print("Loading testset...")
        self.load_test()