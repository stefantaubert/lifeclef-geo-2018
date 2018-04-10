import networkx as nx
import matplotlib.pyplot as plt
import pickle
import data_paths
from species_group import SpeciesGroup
a = [1, 2, 3, 4]
b = [2, 1, 3, 5, 6]
c = [3, 1, 2, 5]
d = [4, 1, 5, 6]
e = [5, 2, 3, 4, 6]
f = [6, 2, 4, 5]

dic = {
    1: [2, 3, 4],
    2: [1, 3, 5, 6],
    3: [1, 2, 5],
    4: [1, 5, 6],
    5: [2, 3, 4, 6],
    6: [2, 4, 5],
    7: [],
}
with open(data_paths.similar_species, 'rb') as f:
    similar_species_loaded = pickle.load(f)
#print(similar_species_loaded)
#dic = similar_species_loaded
groups = SpeciesGroup().get_groups(similar_species_loaded)
print("Count of groups:", len(groups))
print(groups)

group_counts = {}

for group in groups:
    current_len = len(group)
    if current_len in group_counts.keys():
        group_counts[current_len] += 1
    else:
        group_counts[current_len] = 1

print("Group overwiew (count of species: groups):", group_counts)

G=nx.Graph()

for key, value in dic.items():
    G.add_node(key)
    for val in value:
        G.add_edge(key, val)

nx.draw_networkx_labels(G,pos=nx.spring_layout(G))

# print("Nodes of graph: ")
# print(G.nodes())
# print("Edges of graph: ")
# print(G.edges())
# nx.draw(G, node_size=20)
# plt.show()