import pandas as pd
import numpy as np
import data_paths_analysis as data_paths
from tqdm import tqdm
from collections import Counter
import os
import settings_analysis as settings
import math
import pickle
import SimilarSpeciesExtractor
import networkx as nx
import matplotlib.pyplot as plt
import main_preprocessing
import TextPreprocessing
import GroupExtractor

from get_groups import filter_similar_species

def run():
    main_preprocessing.extract_groups()
    GroupAnalysis()._create()
    
class GroupAnalysis():
    def __init__(self):
        csv, species, species_c = TextPreprocessing.load_train()
        self.csv = csv
        self.species = species
        self.species_count = species_c

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
        row_count = len(self.csv.index)
        counter = Counter(self.csv.species_glc_id.values)
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

    def get_named_groups_dict(self, groups):
        result = {}
        for i in range(len(groups)):
            group = groups[i]
            result[i] = group
        return result

    def _create(self):
        print("Create groups...")
        similar_species_dict = SimilarSpeciesExtractor.load()
        similar_species_dict = filter_similar_species(similar_species_dict, settings.min_edge_count)
        G = self.dict_to_graph(similar_species_dict)
        groups = self.get_groups_of_graph(G)
        group_counts = self.get_group_lengths(groups)

        species_propabilities = self.get_species_propabilities()
        #print(species_propabilities)

        #group_probs = self.get_group_propabilities(groups, species_propabilities)
        groups_for_lengths = self.get_groups_for_lengts(groups)

        group_length_probs = self.get_group_length_propabilities(groups_for_lengths, species_propabilities)

        #print(group_length_probs)
        print("Count of groups:", len(groups))
        print("Count of Species in real groups (greater than 1):", str(self.species_count - group_counts[1]))
        biggest_group = np.amax(list(group_counts.keys()))
        print("Biggest group:", biggest_group, "(" + str(group_counts[biggest_group]) + "x)")
        print("Group overview (count of species: groups):", group_counts)
        print("Export diagrams...")

        x = list(group_length_probs.keys())
        y = list(group_length_probs.values())

        fig = plt.figure(figsize=(20, 20))        
        plt.pie(y, labels=x)
        plt.title("Group length summed probabilities (" + str(len(groups)) + " groups for " + str(self.species_count) + " species) @threshold=" + str(settings.threshold))
        plt.savefig(data_paths.group_length_probabilities, bbox_inches='tight')
        #plt.show()
        plt.close()
        plt.close(fig)
        print("Finished (1/2).", data_paths.group_length_probabilities)

        print("Plot network...")
        fig = plt.figure(figsize=(20, 20))   
        plt.title("Groupnetwork (" + str(len(groups)) + " groups for " + str(self.species_count) + " species) @threshold=" + str(settings.threshold))
        #nx.draw_networkx_labels(G,pos=nx.spring_layout(G))
        nx.draw(G, node_size=1)
        plt.savefig(data_paths.group_network, bbox_inches='tight')
        #plt.show()
        plt.close()
        plt.close(fig)
        print("Finished (2/2).", data_paths.group_network)

    def extract(self):
        if not os.path.exists(data_paths.named_groups):
            self._create()
        else:
            print("Groups already exist.")
            

if __name__ == "__main__":
    main_preprocessing.extract_groups()
    GroupAnalysis()._create()