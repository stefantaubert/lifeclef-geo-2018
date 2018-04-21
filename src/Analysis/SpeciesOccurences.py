import module_support_analysis
import pandas as pd
import data_paths_analysis as data_paths
from tqdm import tqdm
import matplotlib.pyplot as plt
import settings_analysis as settings
import numpy as np
from collections import Counter
import main_preprocessing
import os

class SpeciesOccurences():

    def __init__(self):
        self.csv = pd.read_csv(data_paths.train)
        #print("Count of different species:", data.species_count)
        self.species_values = self.csv["species_glc_id"].values
        counter = Counter(self.species_values)
        countRows = len(self.csv.index)
        self.names = []
        self.occurences = []
        self.percents = []
        for item, c in counter.most_common():
            self.names.append(item)
            self.occurences.append(int(c))
            self.percents.append(c / countRows * 100)

    def save_csv(self):
        resulting_rows = list(zip(self.names, self.occurences, self.percents))
        results_array = np.asarray(resulting_rows) #list to array to add to the dataframe as a new column
        result_ser = pd.DataFrame(results_array, columns=["species", "occurences", "percents"])   
        result_ser.to_csv(data_paths.species_occurences, index=False)
        #519 species have >= 100 occurences
        #986 species have < 10 occurences

    def plot_pie(self):      
        plt.figure(figsize=(15,15))
        plt.pie(self.percents)#, labels=names)
        plt.show()

        # ind = np.arange(len(values))  # the x locations for the groups
        # width = 0.5       # the width of the bars
        # _, ax = plt.subplots()
        # ax.bar(ind, values, width, color='r')
        # ax.set_ylabel('Percent')
        # ax.set_xticks(ind+width/2.)
        # ax.set_xticklabels(names)

        plt.show()

def create():
    if not os.path.exists(data_paths.species_occurences):
        main_preprocessing.create_trainset()
        species_occurences = SpeciesOccurences()
        species_occurences.save_csv()
        print("Species occurences saved.")
    else:
        print("Species occurences already saved.")


if __name__ == '__main__':
    create()