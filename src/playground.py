import networkx as nx
import matplotlib.pyplot as plt
import pickle
import data_paths
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
dic = similar_species_loaded
G=nx.Graph()

for key, value in dic.items():
    G.add_node(key)
    for val in value:
        G.add_edge(key, val)


print("Nodes of graph: ")
print(G.nodes())
print("Edges of graph: ")
print(G.edges())
nx.draw(G, node_size=20)
plt.show()