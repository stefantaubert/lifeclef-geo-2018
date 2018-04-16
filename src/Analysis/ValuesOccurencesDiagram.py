import pandas as pd
import data_paths
from tqdm import tqdm
import matplotlib.pyplot as plt
from Data import Data
import settings

class ValuesOccurencesDiagram():
    '''Plots a diagram which shows the occurences of all different values per channel of the complete set. Values are rounded to integer.'''

    def __init__(self, rows, cols, csv):
        self.rows = rows
        self.cols = cols
        self.csv = csv

        drop = ["patch_id", "day", "month", "year"]
        if "species_glc_id" in self.csv.columns.values:
            drop.append("species_glc_id")

        self.csv = self.csv.drop(drop, axis=1) ### Tag usw haben manchmal keine werte
        self.counter = 0

    def plot_data(self, dest_pdf):
        fig = plt.figure(figsize=(24, 13))        
        plt.subplots_adjust(hspace=0.8, wspace=0.4)
        for col in tqdm(self.csv.columns.values):
            self.plot(col)
        
        print("Rendering and saving plot...")
        plt.savefig(dest_pdf, bbox_inches='tight')
        print("Saving completed.")
        plt.show()
        plt.close(fig)

    def plot(self, col_name):
        counts = {}
        
        for _, row in self.csv.iterrows():
            chbio = int(round(row[col_name], settings.round_data_ndigits))
            if chbio not in counts.keys():
                counts[chbio] = 0

            counts[chbio] += 1
            
        #print(counts)

        x = list(counts.keys())
        y = list(counts.values())

        self.counter += 1
        plt.subplot(self.rows, self.cols, self.counter)
        plt.bar(x,y,align='center') # A bar chart
        plt.xlabel(col_name)
        plt.ylabel('occurence')

if __name__ == '__main__':
    data = Data()
    # data.load_train()
    # ValuesOccurencesDiagram(5, 7, data.train).plot_data(data_paths.values_occurences_train)

    data.load_test()
    ValuesOccurencesDiagram(5, 7, data.test).plot_data(data_paths.values_occurences_test)