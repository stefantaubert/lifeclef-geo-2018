import pandas as pd
import data_paths_analysis as data_paths
from tqdm import tqdm
import matplotlib.pyplot as plt
from Data import Data

class SpeciesOccurencesPerValueDiagram():
    '''Plots a diagram which shows the occurences of all different values per channel of the complete set. Values are rounded to integer.'''

    def __init__(self, rows, cols):
        self.data = Data()
        self.data.load_train()
        self.rows = rows
        self.cols = cols
        self.csv = self.data.train

        drop = ["patch_id", "day", "month", "year"]      

        self.csv = self.csv.drop(drop, axis=1) ### Tag usw haben manchmal keine werte
        self.counter = 0

    def plot_data(self):
        fig = plt.figure(figsize=(24, 13))        
        plt.subplots_adjust(hspace=0.8, wspace=0.4)
        for col in tqdm(self.csv.columns.values):
            if col != "species_glc_id":
                self.plot(col)
        
        print("Rendering and saving plot...")
        plt.savefig(data_paths.species_occurences_per_value, bbox_inches='tight')
        print("Saving completed.")
        plt.show()
        plt.close(fig)

    def plot(self, col_name):
        counts = {key: [] for key in set(self.csv[col_name].values)}
       
        for _, row in self.csv.iterrows():
            chbio = float(row[col_name])
            species = int(row["species_glc_id"])
            if species not in counts[chbio]:
                counts[chbio].append(species)
        
        res  = {k: len(v) for k, v in counts.items()}

        #print(res)

        x = list(res.keys())
        y = list(res.values())

        self.counter += 1
        plt.subplot(self.rows, self.cols, self.counter)
        plt.bar(x,y,align='center') # A bar chart
        plt.xlabel(col_name)
        plt.ylabel('c_species')


if __name__ == '__main__':
    SpeciesOccurencesPerValueDiagram(5, 7).plot_data()