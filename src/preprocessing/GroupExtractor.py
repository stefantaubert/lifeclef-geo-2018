import module_support_pre
import pandas as pd
import numpy as np
import data_paths_pre as data_paths
from tqdm import tqdm
from collections import Counter
import os
import settings_preprocessing
import math
import pickle
import SimilarSpeciesExtractor
import networkx as nx
import matplotlib.pyplot as plt

def load():
    assert os.path.exists(data_paths.named_groups)

    with open(data_paths.named_groups, 'rb') as f:
        named_groups = pickle.load(f)
    
    return named_groups

def extract():
    if not os.path.exists(data_paths.named_groups):
        GroupExtractor()._create()
    else: 
        print("Groups already exist.")
            
class GroupExtractor():

    def dict_to_graph(self, dictionary):
        G=nx.Graph()
        
        for key, value in dictionary.items():
            G.add_node(key)
            for val in value:
                G.add_edge(key, val)
        
        return G

    def get_groups_of_graph(self, G):
        groups = []
        processed_nodes = []

        for node in G.nodes():
            if node not in processed_nodes:
                connected_nodes = nx.node_connected_component(G, node )
                groups.append(connected_nodes)
                processed_nodes.extend(connected_nodes)
            
        return groups

    def get_named_groups_dict(self, groups):
        result = {}
        for i in range(len(groups)):
            group = groups[i]
            result[i] = group
        return result

    def _create(self):
        print("Create groups...")
        similar_species_dict = SimilarSpeciesExtractor.load()
       
        G = self.dict_to_graph(similar_species_dict)
        groups = self.get_groups_of_graph(G)
        print(groups)
        print("Save groups to file...")
        group_file = open(data_paths.groups, 'w')
        for item in groups:
            group_file.write("%s\n" % item)

        named_groups_dict = self.get_named_groups_dict(groups)
        pickle.dump(named_groups_dict, open(data_paths.named_groups, 'wb'))
        print("Completed.")


if __name__ == "__main__":
    GroupExtractor()._create()