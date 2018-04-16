import numpy as np
import data_paths
import tifffile
import settings as stg
import itertools as it

def loadImage(sample):
    patch_dir_name = sample[0]
    patch_id = sample[1]
    img = tifffile.imread(data_paths.patch_train+'/{}/patch_{}.tif'.format(patch_dir_name, patch_id))

    return img

def getDatasetChunk(samples):
    for i in range(0, len(samples), stg.BATCH_SIZE):
        yield samples[i:i+stg.BATCH_SIZE]

def getNextImageBatch(samples, species_map):
    for chunk in getDatasetChunk(samples):
        x_batch = np.zeros((stg.BATCH_SIZE, 33, 64, 64), dtype=np.uint8)
        y_batch = np.zeros((stg.BATCH_SIZE, len(species_map.keys())))
        species_ids_batch = np.zeros(stg.BATCH_SIZE)
        current_batch_slot = 0

        for sample in chunk:
            x = loadImage(sample)
            y = np.zeros(len(species_map.keys()))
            y[species_map[sample[2]]] = 1 

            x_batch[current_batch_slot] = x
            y_batch[current_batch_slot] = y
            species_ids_batch[current_batch_slot] = sample[2]
            current_batch_slot += 1

        x_batch = x_batch[:current_batch_slot]
        y_batch = y_batch[:current_batch_slot]
        species_ids_batch = species_ids_batch[:current_batch_slot]

        yield x_batch, y_batch, species_ids_batch

def nextBatch(samples, species_map):
    for x, y, species_ids in it.cycle(getNextImageBatch(samples, species_map)):
        yield (x, y)

def nextValidationBatch(samples, species_map):
    for x, y, species_ids in getNextImageBatch(samples, species_map):
        yield (x, y, species_ids)


#Code für Single Channel Batches


def nextSingleChannelBatch(samples, species_map, channel_index):
    for x, y, species_ids in it.cycle(getNextSingleChannelImageBatch(samples, species_map, channel_index)):
        yield (x, y)

def nextSingleChannelValidationBatch(samples, species_map, channel_index):
    for x, y, species_ids in getNextSingleChannelImageBatch(samples, species_map, channel_index):
        yield (x, y, species_ids)

def getNextSingleChannelImageBatch(samples, species_map, channel_index):
    for chunk in getDatasetChunk(samples):
        x_batch = np.zeros((stg.BATCH_SIZE, 1, 64, 64), dtype=np.uint8)
        y_batch = np.zeros((stg.BATCH_SIZE, len(species_map.keys())))
        species_ids_batch = np.zeros(stg.BATCH_SIZE)
        current_batch_slot = 0

        for sample in chunk:
            x = loadImage(sample)
            y = np.zeros(len(species_map.keys()))
            y[species_map[sample[2]]] = 1 

            x_batch[current_batch_slot] = x[channel_index]
            y_batch[current_batch_slot] = y
            species_ids_batch[current_batch_slot] = sample[2]
            current_batch_slot += 1

        x_batch = x_batch[:current_batch_slot]
        y_batch = y_batch[:current_batch_slot]
        species_ids_batch = species_ids_batch[:current_batch_slot]

        yield x_batch, y_batch, species_ids_batch