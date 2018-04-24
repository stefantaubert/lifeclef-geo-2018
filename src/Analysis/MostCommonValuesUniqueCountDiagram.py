import module_support_analysis
import pandas as pd
import data_paths_analysis as data_paths
from tqdm import tqdm
import matplotlib.pyplot as plt
import MostCommonValueExtractor
import os
import main_preprocessing
import ImageToCSVConverter
import TextPreprocessing
from collections import Counter
import random

class MostCommonValuesUniqueCountDiagram:
    
    def __init__(self):
        ignore_columns = ['occurence', 'species_glc_id']
        
        self.csv = MostCommonValueExtractor.load()
        self.columns = [c for c in self.csv.columns.values if c not in ignore_columns]
    
    def plot(self):
        cols = {col: set([]) for col in self.columns}
        for _, row in self.csv.iterrows():
            for c in cols.keys():
                cols[c].add(float(row[c]))
        
        column_names = list(cols.keys())
        x = [i for i in range(len(cols.keys()))]
        y = [len(l) for l in cols.values()]

        plt.figure()
        plt.bar(x,y,align='center') # A bar chart
        plt.xlabel("channel")
        plt.ylabel('count of unique values')

        plt.savefig(data_paths.most_common_values_unique_count, bbox_inches='tight')
        plt.close()

    def random_search_column_combinations(self, search_count_of_combinations, features_count):
        combinations = []
        result = []
        pbar = tqdm(total=search_count_of_combinations)
        while len(combinations) < search_count_of_combinations:
            pbar.update(1)
            #count = random.randint(0, max_features_count)#len(self.columns) - 1)
            combination = random.sample(self.columns, features_count)
            combination.sort()
            if combination not in combinations:
                combinations.append(combination)
                score = self.get_identified_species_count(combination)
                result.append((score,len(self.csv.index), score/len(self.csv.index), len(combination), combination))
        result.sort()
        result = list(reversed(result))
        pbar.close()
        return result

    def get_identified_species_count(self, columns):
        res = []
        for _, row in self.csv.iterrows():
            current_comb = ""
            for col in columns:
                current_comb += str(row[col])
            res.append(current_comb)
        c = Counter(res)
        return len(c)

    def search_combinations(self):
        print("Searching combinations...")
        result = self.random_search_column_combinations(100, 20)[:5]
        amount, speciescount, score, count, combinations = result[0]
        print(str(len(combinations)), "channels can distinguish", str(round(score * 100, 2)) + "%", "of all species")
        text_file = open(data_paths.most_common_values_best_features, "w")
        text_file.write("Amount: " + str(amount) + "\n")
        text_file.write("Species count: " + str(speciescount) + "\n")
        text_file.write("Score: " + str(score) + "\n")
        text_file.write("Features count: " + str(count) + "\n")
        text_file.write("Features: '" + "', '".join(x for x in combinations) + "'")
        text_file.close()
    
if __name__ == "__main__":
    ImageToCSVConverter.extract_occurences_train()
    TextPreprocessing.extract_train()
    MostCommonValueExtractor.extract()
    MostCommonValuesUniqueCountDiagram().search_combinations()
    #MostCommonValuesUniqueCountDiagram().plot()