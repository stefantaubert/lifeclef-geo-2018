import pandas as pd
import data_paths_analysis as data_paths
from tqdm import tqdm
import matplotlib.pyplot as plt
from Data import Data
import MostCommonValueExtractor

class MostCommonValueDiagram:
    '''Draws a diagram which shows the occurences for the most common values for all species'''
    
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.csv = MostCommonValueExtractor.load()
        self.csv = self.csv.drop(['occurence', 'species_glc_id'], axis=1)
        self.counter = 0

    def plot_data(self):
        # 1920x1080 = 32, 18
        # A4 = 8.3 x 11.7
        fig = plt.figure(figsize=(24, 13))
        plt.subplots_adjust(hspace=0.8, wspace=0.4)
        for col in tqdm(self.csv.columns.values):
            self.plot(col)
        print("Rendering and saving plot...")
        plt.savefig(data_paths.most_common_values_diagram, bbox_inches='tight')
        plt.show()
        plt.close(fig)

    def plot(self, col_name):
        counts = {key: 0 for key in set(self.csv[col_name].values)}
        
        for _, row in self.csv.iterrows():
            chbio = float(row[col_name])
            counts[chbio] += 1
            
        #print(counts)

        x = list(counts.keys())
        y = list(counts.values())

        self.counter += 1
        plt.subplot(self.rows, self.cols, self.counter)
        plt.bar(x,y,align='center') # A bar chart
        plt.xlabel(col_name)
        plt.ylabel('occurence')

if __name__ == "__main__":
    MostCommonValueDiagram(5,7).plot_data()