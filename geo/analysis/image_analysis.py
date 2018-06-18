import numpy as np
import pandas as pd
import tifffile
import pickle
import sys
import seaborn
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

from geo.analysis.data_paths import heatmaps_dir
from geo.data_paths import patch_train
from geo.data_paths import occurrences_train

count_channels = 33
image_dimension = 64
readcount = 50000

def analyse_tiffs(species_id, df):

    #df = pd.read_csv(data_paths.occurrences_train, sep=';', low_memory=False)

    #df = df.head(readcount)

    heatmaps = np.ndarray(shape=(33, 64, 64), dtype=np.float64)

    species_count = 0

    for _, row in tqdm(df.iterrows(), miniters=100):
        current_species_glc_id = row["species_glc_id"]
        if current_species_glc_id == species_id:
            current_patch_dirname = row["patch_dirname"]
            current_patch_id = row["patch_id"]

            img = tifffile.imread(patch_train+'/{}/patch_{}.tif'.format(current_patch_dirname, current_patch_id))
            assert img.shape == (count_channels, image_dimension, image_dimension)

            img = np.array(img, dtype=np.float64)

            species_count = species_count + 1

            for i in range(len(img)):
                heatmaps[i] = heatmaps[i] + (img[i]/255)

    
    print(species_count)
    print(heatmaps)
    heatmaps = heatmaps/species_count
    print(heatmaps)

    for i in range(len(heatmaps)):
        results_dir = heatmaps_dir + 'species{0}/'.format(species_id)
        if not os.path.isdir(results_dir):
            print("created")
            os.makedirs(results_dir)

        seaborn.heatmap(data=heatmaps[i], vmin=0, vmax=1, yticklabels=False, xticklabels=False)
        #print(heatmaps[i])
        #seaborn.plt.show()
        plt.savefig(heatmaps_dir + 'species{0}/heatmap_channel{1}'.format(species_id,i))
        print(heatmaps_dir + 'species{0}/heatmap_channel{1}'.format(species_id,i))
        plt.clf()

if __name__ == '__main__':
    df = pd.read_csv(occurrences_train, sep=';', low_memory=False)
    analyse_tiffs(890, df)
    analyse_tiffs(775, df)
    analyse_tiffs(912, df)