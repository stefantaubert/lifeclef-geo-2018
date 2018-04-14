import numpy as np
import data_paths
import tifffile

def loadImage(sample):
    patch_dir_name = sample[0]
    patch_id = sample[1]
    img = tifffile.imread(data_paths.patch_train+'/{}/patch_{}.tif'.format(patch_dir_name, patch_id))

    return img

def getDatasetChunk(samples):
    for i in range(0, len(samples), 32):
        yield samples[i:i+32]

def getNextImageBatch(samples, species_map):
    for chunk in getDatasetChunk(samples):
        #TODO: replace with Constants from settings
        x_batch = np.zeros((32, 33, 64, 64), dtype=np.uint8)
        y_batch = np.zeros((32, len(species_map.keys())))

        current_batch_slot = 0

        for sample in chunk:
            x = loadImage(sample)
            y = np.zeros(len(species_map.keys()))
            y[species_map[sample[2]]] = 1 

            x_batch[current_batch_slot] = x
            y_batch[current_batch_slot] = y
            current_batch_slot += 1

        x_batch = x_batch[:current_batch_slot]
        y_batch = y_batch[:current_batch_slot]

        yield x_batch, y_batch

def nextBatch(samples, species_map):
    for x, y in getNextImageBatch(samples, species_map):
        yield (x, y)