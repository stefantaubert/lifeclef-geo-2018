import module_support_analysis
import pandas as pd
import data_paths_analysis as data_paths
from tqdm import tqdm
import matplotlib.pyplot as plt
import MostCommonValueExtractor
import os
import main_preprocessing

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
    main_preprocessing.extract_groups()
    plot = [193.0, 2276.0, 1317.0, 487.0, 648.0, 3082.0, 205.0, 238.0, 2093.0, 2416.0, 1423.0, 2323.0, 1556.0, 2356.0, 2294.0, 153.0, 2079.0, 889.0, 383.0]
    d = MostCommonValuesPerSpeciesDiagram()
    for p in plot:
       d.plot(p)
    #MostCommonValuesPerSpeciesDiagram().plot_all_species()