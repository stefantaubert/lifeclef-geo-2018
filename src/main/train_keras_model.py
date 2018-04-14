 
from data_reading import imageList_generator
from data_reading import batch_generator as bg
import data_paths
import pickle
import numpy as np

if __name__ == '__main__':
    samples = np.load(data_paths.train_samples)
    with open(data_paths.train_samples_species_map, 'rb') as f:
        species_map = pickle.load(f)
    
    



