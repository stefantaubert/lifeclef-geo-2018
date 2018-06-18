'''Draws a diagram which shows the occurences for the most common values for all species'''

import matplotlib.pyplot as plt
from tqdm import tqdm

from geo.preprocessing.groups.most_common_value_extraction import load_most_common_values
from geo.preprocessing.preprocessing import extract_groups
from geo.analysis.data_paths import most_common_values_diagram

rows = 5
cols = 7

def plot_most_common_value_diagram():
    extract_groups()
    csv = load_most_common_values()
    csv = csv.drop(['occurence', 'species_glc_id'], axis=1)
    counter = 0
    # 1920x1080 = 32, 18
    # A4 = 8.3 x 11.7
    fig = plt.figure(figsize=(24, 13))
    plt.subplots_adjust(hspace=0.8, wspace=0.4)
    for col in tqdm(csv.columns.values):
        counter += 1
        plot(col, csv, plt, counter)

    print("Rendering and saving plot...")
    plt.savefig(most_common_values_diagram, bbox_inches='tight')
    plt.show()
    plt.close(fig)

def plot(col_name, csv, plt, counter):
    counts = {key: 0 for key in set(csv[col_name].values)}
    
    for _, row in csv.iterrows():
        chbio = float(row[col_name])
        counts[chbio] += 1
        
    x = list(counts.keys())
    y = list(counts.values())

    plt.subplot(rows, cols, counter)
    plt.bar(x,y,align='center') # A bar chart
    plt.xlabel(col_name)
    plt.ylabel('occurence')

if __name__ == "__main__":
    plot_most_common_value_diagram()