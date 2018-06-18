import module_support_main
import numpy as np
import data_paths_main as data_paths
import tifffile
import settings_main as stg
import itertools as it
import cv2
from data_reading.augmentations import flipImage, rotateImage, cropMultiChannelImage, cropSingleChannelImage
from data_reading.utils import loadImage

def resizeImage(image, h, w):
    resized_image = np.zeros(shape=(33, h, w))
    for i in range(image.shape[0]):
        resized_image[i] = cv2.resize(image[i], (h, w), interpolation=cv2.INTER_LINEAR)

    return resized_image

def getDatasetChunk(samples):
    for i in range(0, len(samples), stg.BATCH_SIZE):
        yield samples[i:i+stg.BATCH_SIZE]

def getNextImageBatch(samples, species_map, augment=False):
    for chunk in getDatasetChunk(samples):
        if(stg.resize):
            x_batch = np.zeros((stg.BATCH_SIZE, 33, stg.resize_h, stg.resize_w), dtype=np.uint8)
        else:
            x_batch = np.zeros((stg.BATCH_SIZE, 33, 64, 64), dtype=np.uint8)
        y_batch = np.zeros((stg.BATCH_SIZE, len(species_map.keys())))
        species_ids_batch = np.zeros(stg.BATCH_SIZE)
        glc_ids_batch = np.zeros(stg.BATCH_SIZE)
        current_batch_slot = 0

        for sample in chunk:
            x = loadImage(sample)
            if(augment):
                if np.random.random_sample() > 0.5:
                    x = flipImage(x)
                if np.random.random_sample() > 0.5:
                    x = rotateImage(x, 90)
                if np.random.random_sample() > 0.5:
                    x = cropMultiChannelImage(x)
            if(stg.resize):
                x = resizeImage(x, stg.resize_h, stg.resize_w)
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

def nextBatch(samples, species_map, augment=True):
    for x, y, species_ids, glc_ids in it.cycle(getNextImageBatch(samples, species_map, augment=augment)):
        yield (x, y)

def nextValidationBatch(samples, species_map):
    for x, y, species_ids, glc_ids in getNextImageBatch(samples, species_map):
        yield (x, y, species_ids, glc_ids)

def nextTrainingValidationBatch(samples, species_map):
    for x, y, species_ids, glc_ids in it.repeat(getNextImageBatch(samples, species_map)):
        yield (x, y)

def nextTestBatch(samples, species_map):
    for x, _, _, glc_ids in getNextImageBatch(samples, species_map):
        yield (x, glc_ids)
