import pandas as pd
import data_paths
from tqdm import tqdm
import matplotlib.pyplot as plt
from Data import Data

class ValuesOccurencesDiagram():
    def __init__(self, rows, cols, csv):
        self.rows = rows
        self.cols = cols
        self.csv = csv

        drop = ["patch_id", "day", "month", "year"]
        if "species_glc_id" in self.csv.columns.values:
            drop.append("species_glc_id")

        self.csv = self.csv.drop(drop, axis=1) ### Tag usw haben manchmal keine werte
        self.counter = 0

    def plot_data(self):
        plt.subplots_adjust(hspace=0.8, wspace=0.4)
        for col in self.csv.columns.values:
            self.plot(col)
        plt.show()

    def plot(self, col_name):
        counts = {key: 0 for key in set(self.csv[col_name].values)}
        
        for index, row in tqdm(self.csv.iterrows()):
            chbio = float(row[col_name])
            counts[chbio] += 1
            
        print(counts)

        x = list(counts.keys())
        y = list(counts.values())

        self.counter += 1
        plt.subplot(self.rows, self.cols, self.counter)
        plt.bar(x,y,align='center') # A bar chart
        plt.xlabel(col_name)
        plt.ylabel('occurence')

if __name__ == '__main__':
    data = Data()
    data.load_train()
    ValuesOccurencesDiagram(5, 7, data.train).plot_data()
    #ValuesOccurencesDiagram(5, 7, data.test).plot_data()