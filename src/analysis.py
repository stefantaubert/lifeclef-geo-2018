import pandas as pd
import numpy as np
import data_paths
from collections import Counter
from itertools import repeat, chain
import matplotlib.pyplot as plt
from tqdm import tqdm
from bisect import bisect_left
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

def binary_search(a, x, lo=0, hi=None):  # can't use a to specify default for hi
    hi = hi if hi is not None else len(a)  # hi defaults to len(a)   
    pos = bisect_left(a, x, lo, hi)  # find insertion position
    return (pos if pos != hi and a[pos] == x else -1)  # don't walk off the end

class py_plotter_species_count:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.csv = pd.read_csv(data_paths.occurrences_train_gen)#, nrows=100)
        print(self.csv)
        self.csv = self.csv.drop(["patch_id", "day", "month", "year"], axis=1) ### Tag usw haben manchmal keine werte
        self.counter = 0

    def plot_data(self):
        plt.subplots_adjust(hspace=0.8, wspace=0.4)
        for col in self.csv.columns.values:
            if col != "species_glc_id":
                self.plot(col)
        plt.show()

    def plot(self, col_name):
        counts = {key: [] for key in set(self.csv[col_name].values)}
        
        for index, row in tqdm(self.csv.iterrows()):
            chbio = float(row[col_name])
            species = int(row["species_glc_id"])
            if species not in counts[chbio]:
                counts[chbio].append(species)
        
        res  = {k: len(v) for k, v in counts.items()}

        print(res)

        x = list(res.keys())
        y = list(res.values())

        self.counter += 1
        plt.subplot(self.rows, self.cols, self.counter)
        plt.bar(x,y,align='center') # A bar chart
        plt.xlabel(col_name)
        plt.ylabel('c_species')

class py_plotter_value_count:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.csv = pd.read_csv(data_paths.occurrences_train_gen)#, nrows=100)
        print(self.csv)
        self.csv = self.csv.drop(["patch_id", "day", "month", "year"], axis=1) ### Tag usw haben manchmal keine werte
        self.counter = 0

    def plot_data(self):
        plt.subplots_adjust(hspace=0.8, wspace=0.4)
        for col in self.csv.columns.values:
            if col != "species_glc_id":
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
        plt.ylabel('c_species')

def analyse_spec_occ():
    csv = pd.read_csv(data_paths.occurrences_train_gen, sep=';')
    print("Count of different species:", len(set(csv["species_glc_id"].values)))
    a = csv["species_glc_id"].values

    counter = Counter(a)
    #print(list(chain.from_iterable(repeat(i, c) for i,c in Counter(a).most_common())))

    print(counter.most_common()[:5])
    countRows = len(csv.index)
    res = []
    for item, c in counter.most_common():
        res.append((item, c / countRows * 100))

    print(res[:5])

    sum = 0
    i = 0
    perc = 70
    for item, percent in res:
        i += 1
        sum += percent
        if sum >= perc:
            print("First", i , "items hold >=",perc,"percent of occurences")
            break
    
    data = res[:i]
    names, values = zip(*data)  # @comment by Matthias
    # names = [x[0] for x in data]  # These two lines are equivalent to the the zip-command.
    # values = [x[1] for x in data] # These two lines are equivalent to the the zip-command.

    ind = np.arange(len(data))  # the x locations for the groups
    width = 0.5       # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, values, width, color='r')

    # add some text for labels, title and axes ticks
    ax.set_ylabel('Percent')
    ax.set_xticks(ind+width/2.)
    ax.set_xticklabels(names)

    plt.show()

def analyse_spec():
    csv = pd.read_csv(data_paths.occurrences_train, sep=';')



    species = set(csv["species_glc_id"].values)

    print("Count of different species:", len(set(csv["species"].values)))
    print("Count of different scientificnames:", len(set(csv["scientificname"].values)))
    print("Count of different species ids:", len(species))
    print("Last Species:", list(species)[-1])
    # hier sieht man dass die Speziesanzahl gleich der letzten Spezies entspricht, also alle Spezies lückenlos vorkommen
    print("All species:", species)

    print("Count of rows: ", len(csv.index))
    print("Count of columns: ", len(csv.columns.values))
    print("Column names: ", csv.columns.values)

if __name__ == '__main__':
    #py_plotter_species_count(5, 7).plot_data()
    py_plotter_value_count(5, 7).plot_data()