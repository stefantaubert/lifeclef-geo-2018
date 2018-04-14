import pandas as pd
import numpy as np
import data_paths
from Data import Data
from tqdm import tqdm
from collections import Counter
import os
import math
import SimilarSpeciesExtractor
import networkx as nx
import matplotlib.pyplot as plt

def load():
    assert os.path.exists(data_paths.channel_map_diff)
    
    return pd.read_csv(data_paths.channel_map_diff)

class GroupExtractor():
    def __init__(self):
        self.data = Data()
        self.data._load_train()

    def plot_network(self, G):
        nx.draw_networkx_labels(G,pos=nx.spring_layout(G))
        nx.draw(G, node_size=20)
        plt.show()

    def get_group_lengths(self, groups):
        group_counts = {}

        for group in groups:
            current_len = len(group)
            if current_len in group_counts.keys():
                group_counts[current_len] += 1
            else:
                group_counts[current_len] = 1

        return group_counts

    def get_groups_for_lengts(self, groups):
        result = {}
        for group in groups:
            current_len = len(group)
            if current_len not in result.keys():
                result[current_len] = []
            
            result[current_len].append(group)
        return result

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

    def get_species_propabilities(self):
        row_count = len(self.data.train.index)
        counter = Counter(self.data.train.species_glc_id.values)
        res = {}
        for item, c in counter.most_common():
            res[item] = c / row_count * 100
        return res

    def get_group_length_propabilities(self, groups_lengths, species_probs):
        res = {}
        for length, groups in groups_lengths.items():
            res[length] = 0
            for group in groups:
                for species in group:
                    res[length] += species_probs[species]
        return res 

    def get_group_propabilities(self, groups, species_props):
        res = []
        for group in groups:
            group_prop = 0
            for species in group:
                group_prop += species_props[species]
            res.append(group_prop)
        return res

    def _create(self):
        print("Create groups...")
        similar_species_dict = SimilarSpeciesExtractor.load()
       
        G = self.dict_to_graph(similar_species_dict)
        groups = self.get_groups_of_graph(G)
        group_counts = self.get_group_lengths(groups)

        species_propabilities = self.get_species_propabilities()
        print(species_propabilities)

        group_probs = self.get_group_propabilities(groups, species_propabilities)
        groups_for_lengths = self.get_groups_for_lengts(groups)

        group_length_probs = self.get_group_length_propabilities(groups_for_lengths, species_propabilities)

        #print(group_length_probs)
        print("Count of groups:", len(groups))
        print("Group overwiew (count of species: groups):", group_counts)

        x = list(group_length_probs.keys())
        y = list(group_length_probs.values())

        plt.pie(y, labels=x)
        plt.show()

        print("Plot network...")
        #plot_network(G)


if __name__ == "__main__":
    GroupExtractor()._create()