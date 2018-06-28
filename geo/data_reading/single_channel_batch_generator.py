import module_support_main
import numpy as np
import data_paths_main as data_paths
import tifffile
import settings_main as stg
import itertools as it
import cv2
from augmentations import flipImage, rotateImage, cropMultiChannelImage, cropSingleChannelImage
from utils import loadImage

def getDatasetChunk(samples):
    for i in range(0, len(samples), stg.BATCH_SIZE):
        yield samples[i:i+stg.BATCH_SIZE]

def getNextSingleChannelImageBatch(samples, species_map, channel_index, augment=False):
    for chunk in getDatasetChunk(samples):
        x_batch = np.zeros((stg.BATCH_SIZE, 1, 64, 64), dtype=np.uint8)
        y_batch = np.zeros((stg.BATCH_SIZE, len(species_map.keys())))
        species_ids_batch = np.zeros(stg.BATCH_SIZE)
        glc_ids_batch = np.zeros(stg.BATCH_SIZE)
        current_batch_slot = 0

        
        for sample in chunk:
            x = loadImage(sample)
            x = x[channel_index]
            if(augment):
                if np.random.random_sample() > 0.5:
                    x = flipImage(x)
                if np.random.random_sample() > 0.5:
                    x = rotateImage(x, 90)
                if np.random.random_sample() > 0.5:
                    x = cropSingleChannelImage(x)
            y = np.zeros(len(species_map.keys()))
            y[species_map[sample[2]]] = 1 

            x_batch[current_batch_slot] = x
            y_batch[current_batch_slot] = y
            species_ids_batch[current_batch_slot] = sample[2]
            glc_ids_batch[current_batch_slot] = sample[1]
            current_batch_slot += 1

        x_batch = x_batch[:current_batch_slot]
        y_batch = y_batch[:current_batch_slot]
        species_ids_batch = species_ids_batch[:current_batch_slot]
        glc_ids_batch = glc_ids_batch[:current_batch_slot]

        yield x_batch, y_batch, species_ids_batch, glc_ids_batch

def nextBatch(samples, species_map, channel_index, augment=True):
    for x, y, species_ids, glc_ids in it.cycle(getNextSingleChannelImageBatch(samples, species_map, channel_index, augment=augment)):
        yield (x, y)

def nextValidationBatch(samples, species_map, channel_index):
    for x, y, species_ids, glc_ids in getNextSingleChannelImageBatch(samples, species_map, channel_index):
        yield (x, y, species_ids, glc_ids)

def nextTestBatch(samples, species_map, channel_index):
    for x, _, _, glc_ids in getNextSingleChannelImageBatch(samples, species_map, channel_index):
        yield (x, glc_ids)

