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

class Data:
    '''Loads test and trainset and rounds the values.'''
    
    def load_train(self):
        self.train = pd.read_csv(data_paths.occurrences_train_gen)
       
    def load_test(self):
        self.test = pd.read_csv(data_paths.occurrences_test_gen)        

    def load_datasets(self):
        print("Loading trainset...")
        self.load_train()

        print("Loading testset...")
        self.load_test()