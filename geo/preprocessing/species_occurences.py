import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from collections import Counter

from geo.data_paths import train
from geo.data_paths import species_occurences
from geo.preprocessing.preprocessing import create_trainset

def _create():
    create_trainset()
    csv = pd.read_csv(train)
    #print("Count of different species:", data.species_count)
    species_values = csv["species_glc_id"].values
    counter = Counter(species_values)
    countRows = len(csv.index)
    names = []
    occurences = []
    percents = []
    for item, c in counter.most_common():
        names.append(item)
        occurences.append(int(c))
        percents.append(c / countRows * 100)
    resulting_rows = list(zip(names, occurences, percents))
    results_array = np.asarray(resulting_rows) #list to array to add to the dataframe as a new column
    result_ser = pd.DataFrame(results_array, columns=["species", "occurences", "percents"])   
    result_ser.to_csv(species_occurences, index=False)
    print(species_occurences)
    #519 species have >= 100 occurences
    #986 species have < 10 occurences

def load_species_occurences():
    assert os.path.exists(species_occurences)
    return pd.read_csv(species_occurences)

def extract_species_occurences():
    if not os.path.exists(species_occurences):
        _create()
    else:
        print("Species occurences already saved.")
            
if __name__ == '__main__':
    _create()