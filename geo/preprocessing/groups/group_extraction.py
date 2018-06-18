import pandas as pd
import numpy as np
import os
import math
import pickle
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter

from geo.data_paths import named_groups
from geo.data_paths import groups_txt
from geo.settings import min_edge_count
from geo.preprocessing.groups.filter_similar_species import filter_similar_species
from geo.preprocessing.groups.similar_species_extraction import load_similar_species

def load_named_groups():
    assert os.path.exists(named_groups)

    with open(named_groups, 'rb') as f:
        result = pickle.load(f)
    
    return result

def extract_named_groups():
    if not os.path.exists(named_groups):
        _create()
    else: 
        print("Groups already exist.")
            
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
            connected_nodes = nx.node_connected_component(G, node)
            groups.append(connected_nodes)
            processed_nodes.extend(connected_nodes)
    return groups

def get_named_groups_dict(groups):
    result = {}
    for i in range(len(groups)):
        group = groups[i]
        result[i] = group
    return result

def _create():
    print("Create groups...")
    similar_species_dict = load_similar_species()
    similar_species_dict = filter_similar_species(similar_species_dict, min_edge_count)
    G = dict_to_graph(similar_species_dict)
    groups = get_groups_of_graph(G)
    
    print("Save groups to file...")
    group_file = open(groups_txt, 'w')
    for item in groups:
        group_file.write("%s\n" % item)

    named_groups_dict = get_named_groups_dict(groups)
    pickle.dump(named_groups_dict, open(named_groups, 'wb'))
    print("Completed.", named_groups)


if __name__ == "__main__":
    _create()