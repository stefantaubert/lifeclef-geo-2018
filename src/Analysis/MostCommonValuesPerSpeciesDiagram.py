import pandas as pd
import data_paths
from tqdm import tqdm
import matplotlib.pyplot as plt
from Data import Data
import MostCommonValueExtractor
import os

class MostCommonValuesPerSpeciesDiagram:
    '''Draws a diagram which shows of an species the most common value for each channel.'''
    
    def __init__(self):
        self.csv = MostCommonValueExtractor.load()
        self.create_output_dir_if_not_exists()

    def create_output_dir_if_not_exists(self):
        if not os.path.exists(data_paths.species_channel_map_dir):
            os.makedirs(data_paths.species_channel_map_dir)

    def plot_all_species(self):
        species = self.csv['species_glc_id']
        for specie in tqdm(species):
            self.plot(int(specie))
            plt.close()

    def plot(self, specie):
        specie_csv = self.csv[self.csv['species_glc_id'] == specie]
        assert len(specie_csv.index) == 1

        counts = {}
        occ = 0
        ignore_columns = ['occurence', 'species_glc_id']
        columns = specie_csv.columns.values
        species_row = specie_csv.iloc[[0]].iloc[0]

        for i in range(len(columns)):
            current_column = columns[i]
            if current_column not in ignore_columns:    
                counts[i] = species_row[current_column]
        
        occurence = str(species_row["occurence"])
        
        x = list(counts.keys())
        y = list(counts.values())

        plt.figure()
        plt.bar(x,y,align='center') # A bar chart
        plt.xlabel("channel")
        plt.ylabel('occurence')
        plt.title("Species: " + str(specie) + ", occurence: " + occurence)

        plt.savefig(data_paths.species_channel_map_dir + str(specie) + ".png", bbox_inches='tight')
        return plt
if __name__ == "__main__":
    #MostCommonValuesPerSpeciesDiagram().plot(775).show()
    MostCommonValuesPerSpeciesDiagram().plot_all_species()