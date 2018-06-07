import networkx as nx
from tqdm import tqdm
import copy

def filter_similar_species(similar_spec, k):
    similar_species = copy.deepcopy(similar_spec)
    G=nx.Graph()
        
    for key, value in similar_species.items():
        G.add_node(key)
        for val in value:
            G.add_edge(key, val)

    processed_nodes = []

    for node in G.nodes():
        if node not in processed_nodes:
            connected_nodes = nx.node_connected_component(G, node)
            # example [1,2,3,4,5,9]
            while(True):
                removed_sth = False
                for potential_group_item in connected_nodes:
                    count_edges = len(similar_species[potential_group_item])
                    if count_edges < k and count_edges > 0:
                        for node in similar_species[potential_group_item]:
                            assert node in connected_nodes # all connected items are in a group
                            assert potential_group_item in similar_species[node] # if a species is similar to an other this one is also similar to that species
                            similar_species[node].remove(potential_group_item)
                            removed_sth = True
                        similar_species[potential_group_item] = []
                if not removed_sth:
                    break
            processed_nodes.extend(connected_nodes)
    return similar_species