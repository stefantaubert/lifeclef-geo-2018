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
    plot = [897.0, 133.0, 1158.0, 1288.0, 270.0, 783.0, 18.0, 1942.0, 23.0, 1434.0, 801.0, 1447.0, 1576.0, 1454.0, 1790.0, 1075.0, 698.0, 1338.0, 1979.0, 706.0, 69.0, 72.0, 469.0, 220.0, 866.0, 483.0, 354.0, 1511.0, 628.0, 1268.0, 247.0, 1273.0, 379.0, 766.0]
    d = MostCommonValuesPerSpeciesDiagram()
    for p in plot:
       d.plot(p)
    #MostCommonValuesPerSpeciesDiagram().plot_all_species()