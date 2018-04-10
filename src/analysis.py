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
import math
import pickle
import networkx as nx
import matplotlib.pyplot as plt

def binary_search(a, x, lo=0, hi=None):  # can't use a to specify default for hi
    hi = hi if hi is not None else len(a)  # hi defaults to len(a)   
    pos = bisect_left(a, x, lo, hi)  # find insertion position
    return (pos if pos != hi and a[pos] == x else -1)  # don't walk off the end

class channelmap_vector:
    def __init__(self):
        self.csv = pd.read_csv(data_paths.max_values_species)
        y_ids = np.load(data_paths.y_ids)
        species = np.unique(y_ids)
        cols_to_consider = ['chbio_1', 'chbio_2', 'chbio_3', 'chbio_4', 'chbio_5', 'chbio_6',
            'chbio_7', 'chbio_8', 'chbio_9', 'chbio_10', 'chbio_11', 'chbio_12',
            'chbio_13', 'chbio_14', 'chbio_15', 'chbio_16', 'chbio_17', 'chbio_18',
            'chbio_19', 'etp', 'alti',
            'awc_top', 'bs_top', 'cec_top', 'crusting', 'dgh', 'dimp', 'erodi', 'oc_top', 'pd_top', 'text',
            'proxi_eau_fast', 'clc', 
            #'day', 'month', 'year',
            'latitude', 'longitude']
        self.csv = self.csv.drop(["occurence", "species_glc_id"], axis=1)
        
        array = []

        array = self.csv.as_matrix()

        # for specie in tqdm(species):
        #     specie_csv = self.csv[self.csv['species_glc_id'] == specie]
        #     assert len(specie_csv.index) == 1

        #     current_row = []
        #     for index, row in tqdm(specie_csv.iterrows()):
        #         for col in cols_to_consider:
        #             current_row.append(row[col])

        #     array.append(current_row)
        threshold = 40
        similar_species = {k: [] for k in species}
        matrix = []
        for i in tqdm(range(len(array))):
            current_channel_map = np.array(array[i])
            species_distances = []

            for j in range(len(array)):
                is_current_channel_map = j == i
                if is_current_channel_map:
                    species_distances.append(0)
                else:
                    other_channel_map = np.array(array[j])
                    diff_vector = other_channel_map - current_channel_map
                    # betrag des Vektors ausrechnen
                    sum = 0
                    for num in diff_vector:
                        sum += num * num
                    assert len(diff_vector) == len(cols_to_consider)
                    distance = math.sqrt(sum)
                    species_distances.append(distance)
                    if distance <= threshold:
                        current_species = i + 1
                        other_species = j + 1
                        similar_species[current_species].append(other_species)

            if False:
                tmp = species_distances
                tmp.sort()
                print(tmp[:5])
                print(tmp[-5:])
                x = species
                y = species_distances

                plt.figure()
                plt.bar(x,y,align='center') # A bar chart
                plt.xlabel("species")
                plt.ylabel('dist')
                plt.show()
                #plt.savefig(data_paths.species_channel_map_dir + str(specie) + ".png", bbox_inches='tight')

            matrix.append(species_distances)
        
        #print(similar_species)
        pickle.dump(similar_species, open(data_paths.similar_species, 'wb'))

       
        # with open(data_paths.similar_species, 'rb') as f:
        #     similar_species_loaded = pickle.load(f)
        # print(similar_species_loaded)

        #print(result)
        results_array = np.asarray(matrix) #list to array to add to the dataframe as a new column
        result_ser = pd.DataFrame(results_array, columns=species)
        result_ser.to_csv(data_paths.channel_map_diff, index=False)

        G=nx.Graph()

        for key, value in similar_species.items():
            G.add_node(key)
            for val in value:
                G.add_edge(key, val)

        nx.draw(G, node_size=20)
        plt.show()

class csv_species_map:
    def __init__(self):
        self.csv = pd.read_csv(data_paths.max_values_species)
        y_ids = np.load(data_paths.y_ids)
        species = np.unique(y_ids)
        cols_to_consider = ['chbio_1', 'chbio_2', 'chbio_3', 'chbio_4', 'chbio_5', 'chbio_6',
            'chbio_7', 'chbio_8', 'chbio_9', 'chbio_10', 'chbio_11', 'chbio_12',
            'chbio_13', 'chbio_14', 'chbio_15', 'chbio_16', 'chbio_17', 'chbio_18',
            'chbio_19', 'etp', 'alti',
            'awc_top', 'bs_top', 'cec_top', 'crusting', 'dgh', 'dimp', 'erodi', 'oc_top', 'pd_top', 'text',
            'proxi_eau_fast', 'clc', 
            #'day', 'month', 'year',
            'latitude', 'longitude']

        for specie in tqdm(species):
            specie_csv = self.csv[self.csv['species_glc_id'] == specie]
            assert len(specie_csv.index) == 1

            counts = {}
            occ = 0
            for index, row in tqdm(specie_csv.iterrows()):
                i = 0
                for col in cols_to_consider:
                    counts[i] = row[col]
                    i += 1     
                occ = row["occurence"]

            x = list(counts.keys())
            y = list(counts.values())

            plt.figure()
            plt.bar(x,y,align='center') # A bar chart
            plt.xlabel("channel")
            plt.ylabel('occurence')
            plt.title("Species: " + str(specie) + ", occurence: " + str(occ))
            plt.savefig(data_paths.species_channel_map_dir + str(specie) + ".png", bbox_inches='tight')

class csv_max_values_occurrences:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.csv = pd.read_csv(data_paths.max_values_species)
        self.csv = self.csv.drop(['occurence', 'species_glc_id'], axis=1)
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

class csv_max_values_per_species:
    def __init__(self):
        csv = pd.read_csv(data_paths.occurrences_train_gen)
        y_ids = np.load(data_paths.y_ids)
        #print(self.csv)
        species = np.unique(y_ids)
        #csv = csv.drop(["patch_id", "day", "month", "year"], axis=1) ### Tag usw haben manchmal keine werte
        rows_to_consider = ['chbio_1', 'chbio_2', 'chbio_3', 'chbio_4', 'chbio_5', 'chbio_6',
            'chbio_7', 'chbio_8', 'chbio_9', 'chbio_10', 'chbio_11', 'chbio_12',
            'chbio_13', 'chbio_14', 'chbio_15', 'chbio_16', 'chbio_17', 'chbio_18',
            'chbio_19', 'etp', 'alti',
            'awc_top', 'bs_top', 'cec_top', 'crusting', 'dgh', 'dimp', 'erodi', 'oc_top', 'pd_top', 'text',
            'proxi_eau_fast', 'clc', 
            #'day', 'month', 'year',
             'latitude', 'longitude']

        result_cols = rows_to_consider + ['occurence', 'species_glc_id']
            
        #columns = csv.columns.values
        resulting_rows = []
        for specie in tqdm(species):
            specie_csv = csv[csv["species_glc_id"] == specie]
            row = []
            for col in rows_to_consider:
                c = Counter(specie_csv[col])
                most_common_value, occ = c.most_common(1)[0]
                row.append(most_common_value)
            row.append(len(specie_csv.index))
            row.append(specie)
            resulting_rows.append(row)

        
        print("Write data...")
        results_array = np.asarray(resulting_rows) #list to array to add to the dataframe as a new column

        result_ser = pd.DataFrame(results_array, columns=result_cols)
        result_ser.to_csv(data_paths.max_values_species, index=False)
                
class py_species_channels_relative:
    def __init__(self, rows, cols, species_id):
        self.rows = rows
        self.cols = cols
        self.species_id = species_id
        self.csv = pd.read_csv(data_paths.occurrences_train_gen)
        #print(self.csv)
        self.csv = self.csv.drop(["patch_id", "day", "month", "year"], axis=1) ### Tag usw haben manchmal keine werte
        self.csv = self.csv[self.csv["species_glc_id"] == self.species_id]
        self.row_count = len(self.csv.index)
        print(self.csv)
        self.counter = 0

    def plot_data(self):
        plt.subplots_adjust(hspace=0.8, wspace=0.4)
        for col in self.csv.columns.values:
            if col != "species_glc_id":
                self.plot(col)
        
        plt.show()

    def plot(self, col_name):      
        occurences = {key: 0 for key in set(self.csv[col_name].values)}
        
        for index, row in tqdm(self.csv.iterrows()):
            chbio = float(row[col_name])
            occurences[chbio] += 1

        print(occurences)

        y = [val / self.row_count for val in occurences.values()]

        x = list(occurences.keys())
        #y = list(occurences.values()) / self.row_count

        self.counter += 1
        plt.subplot(self.rows, self.cols, self.counter)
        plt.bar(x,y,align='center') # A bar chart
        plt.xlabel(col_name)
        plt.ylabel('percent')
        if self.counter == 4:
            plt.title("Channels for species_glc_id: " + str(self.species_id))

class py_species_channels_absolute:
    def __init__(self, rows, cols, species_id):
        self.rows = rows
        self.cols = cols
        self.species_id = species_id
        self.csv = pd.read_csv(data_paths.occurrences_train_gen)
        #print(self.csv)
        self.csv = self.csv.drop(["patch_id", "day", "month", "year"], axis=1) ### Tag usw haben manchmal keine werte
        self.csv = self.csv[self.csv["species_glc_id"] == self.species_id]
        print(self.csv)
        self.counter = 0

    def plot_data(self):
        plt.subplots_adjust(hspace=0.8, wspace=0.4)
        for col in self.csv.columns.values:
            if col != "species_glc_id":
                self.plot(col)
        
        plt.show()

    def plot(self, col_name):      
        occurences = {key: 0 for key in set(self.csv[col_name].values)}
        
        for index, row in tqdm(self.csv.iterrows()):
            chbio = float(row[col_name])
            occurences[chbio] += 1

        print(occurences)

        x = list(occurences.keys())
        y = list(occurences.values())

        self.counter += 1
        plt.subplot(self.rows, self.cols, self.counter)
        plt.bar(x,y,align='center') # A bar chart
        plt.xlabel(col_name)
        plt.ylabel('occurence')
        if self.counter == 4:
            plt.title("Channels for species_glc_id: " + str(self.species_id))

class py_plotter_combined:
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
        occurences = {key: 0 for key in set(self.csv[col_name].values)}
        
        for index, row in tqdm(self.csv.iterrows()):
            chbio = float(row[col_name])
            species = int(row["species_glc_id"])
            occurences[chbio] += 1
            if species not in counts[chbio]:
                counts[chbio].append(species)
        
        res  = {k: len(v) / occurences[k] for k, v in counts.items()}

        print(res)

        x = list(res.keys())
        y = list(res.values())

        self.counter += 1
        plt.subplot(self.rows, self.cols, self.counter)
        plt.bar(x,y,align='center') # A bar chart
        plt.xlabel(col_name)
        #plt.ylabel('c_species')


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
        self.csv = self.csv.drop(["patch_id", "species_glc_id", "day", "month", "year"], axis=1) ### Tag usw haben manchmal keine werte
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
    #py_plotter_species_count(5, 7).plot_data()
    #py_plotter_value_count(5, 7).plot_data()
    #py_plotter_combined(5, 7).plot_data()
    #analyse_spec_occ()
    #py_species_channels_relative(5,7,890).plot_data()
    #csv_max_values_per_species()
    #csv_max_values_occurrences(5,7).plot_data()
    #csv_species_map()
    channelmap_vector()