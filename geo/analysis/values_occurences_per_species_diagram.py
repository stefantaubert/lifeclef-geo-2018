import module_support_analysis
import pandas as pd
import data_paths_analysis as data_paths
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import main_preprocessing

class ValuesOccurencesPerSpeciesDiagram:
    '''Plots a diagram for one species which shows the occurences for all values of each channel.'''
    
    def __init__(self):
        self.csv = pd.read_csv(data_paths.train)
        self.all_species = self.csv["species_glc_id"]
        self.rows = 5
        self.cols = 7
        self.counter = 0
        self.create_output_dir_if_not_exists()

    def create_output_dir_if_not_exists(self):
        if not os.path.exists(data_paths.value_occurences_species_dir):
            os.makedirs(data_paths.value_occurences_species_dir)

    # def plot_all_species(self):
    #     #Works only for one species
    #     for species in tqdm(self.all_species):
    #         plt = self.plot_species(int(species))
    #         plt.close()

    def plot_species(self, species):
        specie_csv = self.csv[self.csv['species_glc_id'] == species]
        self.occurence = len(specie_csv.index)
        self.species_id  = species
        plt.figure(figsize=(24, 13))        
        plt.subplots_adjust(hspace=0.8, wspace=0.4)
        ignore_columns = ["species_glc_id", "patch_id", "day", "month", "year"]  ### Tag usw haben manchmal keine werte
        for col in tqdm(specie_csv.columns.values):
            if col not in ignore_columns:
                self.plot(specie_csv, col)
        print("Rendering and saving plot...")
        file_name = data_paths.value_occurences_species_dir + str(species) + ".pdf"
        plt.savefig(file_name, bbox_inches='tight')
        print("Saving completed.", file_name)

        return plt

    def plot(self, specie_csv, col_name):      
        occurences = {key: 0 for key in set(specie_csv[col_name].values)}
        
        for _, row in specie_csv.iterrows():
            chbio = float(row[col_name])
            occurences[chbio] += 1

        #print(occurences)

        x = list(occurences.keys())
        y = list(occurences.values())
        most_common_occ = max(y)
        index_mco = y.index(most_common_occ)
        most_common_value = x[index_mco]

        self.counter += 1
        plt.subplot(self.rows, self.cols, self.counter)
        plt.bar(x,y,align='center') # A bar chart
        plt.xlabel(col_name + " - " + str(most_common_value))
        plt.ylabel('occurence')
        if self.counter == 4:
            plt.title("Channels for species_glc_id: " + str(self.species_id) + ", Occurence: " + str(self.occurence))

if __name__ == "__main__":
    main_preprocessing.create_trainset()
    
    ValuesOccurencesPerSpeciesDiagram().plot_species(890)