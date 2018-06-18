'''Draws a diagram which shows of an species the most common value for each channel.'''

import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

from geo.analysis.data_paths import species_channel_map_dir
from geo.preprocessing.groups.most_common_value_extraction import load_most_common_values
from geo.preprocessing.preprocessing import extract_groups

def plot_most_common_values_of_each_species():
    extract_groups()
    csv = load_most_common_values()
    species = csv['species_glc_id']
    plot_most_common_values_of(species)

def plot_most_common_values_of(species):
    extract_groups()
    csv = load_most_common_values()
    for specie in tqdm(species):
        _plot(int(specie), csv)
        plt.close()

def _plot(specie, csv):
    specie_csv = csv[csv['species_glc_id'] == specie]
    assert len(specie_csv.index) == 1

    counts = {}
    ignore_columns = ['occurence', 'species_glc_id']
    columns = specie_csv.columns.values
    species_row = specie_csv.iloc[[0]].iloc[0]

    for i in range(len(columns)):
        current_column = columns[i]
        if current_column not in ignore_columns:    
            counts[i] = species_row[current_column]
    
    occurence = str(species_row["occurence"])
    
    x = list(counts.keys())
    y = list(counts.values())

    plt.figure()
    plt.bar(x,y,align='center') # A bar chart
    plt.xlabel("channel")
    plt.ylabel('occurence')
    plt.title("Species: " + str(specie) + ", occurence: " + occurence)

    out_path = species_channel_map_dir + str(specie) + ".png"
    plt.savefig(out_path, bbox_inches='tight')
    print(out_path)
    return plt
        
if __name__ == "__main__":
    plot = [897.0, 133.0, 1158.0, 1288.0, 1417.0, 270.0, 766.0, 18.0, 23.0]
    plot_most_common_values_of(plot)
    #plot_most_common_values_of_each_species()