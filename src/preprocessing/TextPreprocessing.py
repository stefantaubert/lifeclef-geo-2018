import ImageToCSVConverter
import numpy as np
import pandas as pd
import data_paths
from tqdm import tqdm
import pickle
import settings
import sys
from tqdm import tqdm
from collections import Counter
from bisect import bisect_left
import os

def binary_search(a, x, lo=0, hi=None):  # can't use a to specify default for hi
    hi = hi if hi is not None else len(a)  # hi defaults to len(a)   
    pos = bisect_left(a, x, lo, hi)  # find insertion position
    return (pos if pos != hi and a[pos] == x else -1)  # don't walk off the end

def load_train():
    assert os.path.exists(data_paths.train)

    csv = pd.read_csv(data_paths.train)
    
    species = sorted(csv.species_glc_id.unique())
    species_count = len(species)

    return (csv, species, species_count)

def load_test():
    assert os.path.exists(data_paths.test)
    
    csv = pd.read_csv(data_paths.test)
    return csv

def extract_train():
    if not os.path.exists(data_paths.train):
        TextPreprocessing()._create_train()
    else:
        print("Trainset already exists.")

def extract_test():
    if not os.path.exists(data_paths.test):
        TextPreprocessing()._create_test()
    else: 
        print("Testset already exists.")

class TextPreprocessing:
    def __init__(self):
        self.create_output_dir_if_not_exists()

    def create_output_dir_if_not_exists(self):
        if not os.path.exists(data_paths.preprocessing_dir):
            os.makedirs(data_paths.preprocessing_dir)

    def _create_train(self):
        df = ImageToCSVConverter.load_occurences_train()
        df = df.round(settings.round_data_ndigits)
        df = self.remove_occurences_df(df, settings.min_occurence)
        df.to_csv(data_paths.train, index=False)
   
    def _create_test(self):
        df = ImageToCSVConverter.load_occurences_test()
        df = df.round(settings.round_data_ndigits)
        df.to_csv(data_paths.test, index=False)

    def remove_occurences_df(self, df, occ):
        print("Filter low frequent species...")
        counter = Counter(df.species_glc_id.values)
        ignore_species = []
        for species, count in counter.most_common():
            if count < settings.min_occurence:
                ignore_species.append(species)
        ignore_species.sort()

        remove_indices = []

        for index, row in tqdm(df.iterrows()):
            res = binary_search(ignore_species, row["species_glc_id"])
            if res != -1:
                remove_indices.append(index)

        df = df.drop(remove_indices)
        
        # for index, row in tqdm(self.train.iterrows()):
        #     res = binary_search(ignore_species, row["species_glc_id"])
        #     assert res == -1

        print("Ignored species:", len(ignore_species))

        # self.species = sorted(df.species_glc_id.unique())
        # self.species_count = len(self.species)

        # r = [i + 1 for i in range(len(self.species))]
        # assert self.species == r
        return df

if __name__ == '__main__':
    extract_train()
    extract_test()
