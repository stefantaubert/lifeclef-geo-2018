import numpy as np
import pandas as pd
import data_paths
from tqdm import tqdm
import tifffile
import pickle
import settings
import sys
from tqdm import tqdm
from collections import Counter
from bisect import bisect_left

def binary_search(a, x, lo=0, hi=None):  # can't use a to specify default for hi
    hi = hi if hi is not None else len(a)  # hi defaults to len(a)   
    pos = bisect_left(a, x, lo, hi)  # find insertion position
    return (pos if pos != hi and a[pos] == x else -1)  # don't walk off the end

class Data:
    '''Loads test and trainset and rounds the values.'''
    
    def load_train(self):
        self.train = pd.read_csv(data_paths.occurrences_train_gen)
        self.train = self.train.round(settings.round_data_ndigits)
        print("Filter low frequent species...")
        counter = Counter(self.train.species_glc_id.values)
        ignore_species = []
        for species, count in counter.most_common():
            if count < settings.min_occurence:
                ignore_species.append(species)
        ignore_species.sort()

        remove_indices = []

        for index, row in tqdm(self.train.iterrows()):
            res = binary_search(ignore_species, row["species_glc_id"])
            if res != -1:
                remove_indices.append(index)

        self.train = self.train.drop(remove_indices)
        
        # for index, row in tqdm(self.train.iterrows()):
        #     res = binary_search(ignore_species, row["species_glc_id"])
        #     assert res == -1

        print("Ignored species:", len(ignore_species))

        self.species = sorted(self.train.species_glc_id.unique())
        self.species_count = len(self.species)

        # r = [i + 1 for i in range(len(self.species))]
        # assert self.species == r

    def load_test(self):
        self.test = pd.read_csv(data_paths.occurrences_test_gen)
        self.test = self.test.round(settings.round_data_ndigits)

    def load_datasets(self):
        print("Loading trainset...")
        self.load_train()

        print("Loading testset...")
        self.load_test()