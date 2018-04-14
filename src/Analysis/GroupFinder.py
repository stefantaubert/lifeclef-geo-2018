import pandas as pd
import numpy as np
import data_paths
from Data import Data
from tqdm import tqdm
from collections import Counter
import math
import pickle
import networkx as nx
import matplotlib.pyplot as plt

def get_most_common_value_matrix_df():
    result_cols = cols_to_consider + ['occurence', 'species_glc_id']
    #columns = csv.columns.values
    resulting_rows = []

    for specie in tqdm(data.species):
        specie_csv = data.train[data.train["species_glc_id"] == specie]
        row = []

        for col in cols_to_consider:
            c = Counter(specie_csv[col])
            most_common_value, occ = c.most_common(1)[0]
            row.append(most_common_value)

        row.append(len(specie_csv.index))
        row.append(specie)
        resulting_rows.append(row)

    results_array = np.asarray(resulting_rows) #list to array to add to the dataframe as a new column
    result_ser = pd.DataFrame(results_array, columns=result_cols)   

    return result_ser

def get_vector_length(v):
    summ = 0

    for num in v:
        summ += num * num

    distance = math.sqrt(summ)

    return distance

def get_species_diff_matrix_df(most_common_value_matrix_df):
    most_common_value_matrix.drop(['occurence', 'species_glc_id'], axis=1, inplace=True)
    array = most_common_value_matrix_df.as_matrix()
    assert len(array) == data.species_count
    assert len(most_common_value_matrix.columns.values) == len(cols_to_consider)
    
    matrix = []

    for i in tqdm(range(data.species_count)):
        current_channel_map = np.array(array[i])
        species_distances = []

        for j in range(data.species_count):
            is_current_channel_map = j == i

            if is_current_channel_map:
                species_distances.append(0)
            else:
                other_channel_map = np.array(array[j])
                diff_vector = other_channel_map - current_channel_map
                # betrag des Vektors ausrechnen
                distance = get_vector_length(diff_vector)
                species_distances.append(distance)

        matrix.append(species_distances)
    
    results_array = np.asarray(matrix) #list to array to add to the dataframe as a new column
    result_ser = pd.DataFrame(results_array, columns=data.species)

    return result_ser
  
def get_similar_species_dict(species_diff_matrix_df, threshold):
    assert len(species_diff_matrix_df.index) == data.species_count
    assert len(species_diff_matrix_df.columns.values) == data.species_count
    array = species_diff_matrix_df.as_matrix()
    
    similar_species = {k: [] for k in data.species}

    for i in tqdm(range(data.species_count)):
        for j in range(data.species_count):
            is_current_species = j == i      

            if not is_current_species:
                distance = array[i][j]

                if distance <= threshold:
                    current_species = i + 1
                    other_species = j + 1
                    similar_species[current_species].append(other_species)
    
    return similar_species

def plot_network(G):
    nx.draw_networkx_labels(G,pos=nx.spring_layout(G))
    nx.draw(G, node_size=20)
    plt.show()

def get_group_lengths(groups):
    group_counts = {}

    for group in groups:
        current_len = len(group)
        if current_len in group_counts.keys():
            group_counts[current_len] += 1
        else:
            group_counts[current_len] = 1

    return group_counts

def get_groups_for_lengts(groups):
    result = {}
    for group in groups:
        current_len = len(group)
        if current_len not in result.keys():
            result[current_len] = []
        
        result[current_len].append(group)
    return result

def dict_to_graph(dictionary):
    G=nx.Graph()
    
    for key, value in dictionary.items():
        G.add_node(key)
        for val in value:
            G.add_edge(key, val)
    
    return G

def get_groups_of_graph(G):
    groups = []
    processed_nodes = []

    for node in G.nodes():
        if node not in processed_nodes:
            connected_nodes = nx.node_connected_component(G, node )
            groups.append(connected_nodes)
            processed_nodes.extend(connected_nodes)
        
    return groups

def get_species_propabilities():
    row_count = len(data.train.index)
    counter = Counter(data.train.species_glc_id.values)
    res = {}
    for item, c in counter.most_common():
        res[item] = c / row_count * 100
    return res

def get_group_length_propabilities(groups_lengths, species_probs):
    res = {}
    for length, groups in groups_lengths.items():
        res[length] = 0
        for group in groups:
            for species in group:
                res[length] += species_probs[species]
    return res 

def get_group_propabilities(groups, species_props):
    res = []
    for group in groups:
        group_prop = 0
        for species in group:
            group_prop += species_props[species]
        res.append(group_prop)
    return res

data = Data()
data.load_train()

species_propabilities = get_species_propabilities()
print(species_propabilities)

cols_to_consider = ['chbio_1', 'chbio_2', 'chbio_3', 'chbio_4', 'chbio_5', 'chbio_6',
            'chbio_7', 'chbio_8', 'chbio_9', 'chbio_10', 'chbio_11', 'chbio_12',
            'chbio_13', 'chbio_14', 'chbio_15', 'chbio_16', 'chbio_17', 'chbio_18',
            'chbio_19', 'etp', 'alti',
            'awc_top', 'bs_top', 'cec_top', 'crusting', 'dgh', 'dimp', 'erodi', 'oc_top', 'pd_top', 'text',
            'proxi_eau_fast', 'clc', 
            #'day', 'month', 'year',
            'latitude', 'longitude']

print("Getting most common values...")
if False:
    most_common_value_matrix = get_most_common_value_matrix_df()
    most_common_value_matrix.to_csv(data_paths.max_values_species, index=False)
else: 
    most_common_value_matrix = pd.read_csv(data_paths.max_values_species)

print("Calculate species distances...")
if False:
    species_diff_matrix = get_species_diff_matrix_df(most_common_value_matrix)
    species_diff_matrix.to_csv(data_paths.channel_map_diff, index=False)
else:
    species_diff_matrix = pd.read_csv(data_paths.channel_map_diff)

print("Get similar species...")
if True:
    threshold = 20
    similar_species_dict = get_similar_species_dict(species_diff_matrix, threshold)
    pickle.dump(similar_species_dict, open(data_paths.similar_species, 'wb'))
else:
    with open(data_paths.similar_species, 'rb') as f:
        similar_species_dict = pickle.load(f)

print("Create groups...")
G = dict_to_graph(similar_species_dict)
groups = get_groups_of_graph(G)
group_counts = get_group_lengths(groups)

group_probs = get_group_propabilities(groups, species_propabilities)
groups_for_lengths = get_groups_for_lengts(groups)

group_length_probs = get_group_length_propabilities(groups_for_lengths, species_propabilities)

#print(group_length_probs)
print("Count of groups:", len(groups))
print("Group overwiew (count of species: groups):", group_counts)

x = list(group_length_probs.keys())
y = list(group_length_probs.values())

plt.pie(y, labels=x)
plt.show()

print("Plot network...")
#plot_network(G)
