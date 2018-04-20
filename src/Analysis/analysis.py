import pandas as pd
import numpy as np
import data_paths_analysis as data_paths
from collections import Counter
from itertools import repeat, chain
import matplotlib.pyplot as plt
from tqdm import tqdm
from bisect import bisect_left
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import math
import pickle
import networkx as nx
import matplotlib.pyplot as plt

def analyse_spec_occ():
    csv = pd.read_csv(data_paths.occurrences_train_gen)
    print("Count of different species:", len(set(csv["species_glc_id"].values)))
    a = csv["species_glc_id"].values

    counter = Counter(a)
    #print(list(chain.from_iterable(repeat(i, c) for i,c in Counter(a).most_common())))

    print(counter.most_common()[-3000:-2500])
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
    ###(890.0, 2259), (775.0, 2090), (912.0, 2078), (956.0, 1838), (981.0, 1808)
    analyse_spec()