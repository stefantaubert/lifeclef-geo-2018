import numpy as np
import tifffile
import itertools as it
import cv2
from geo.models.settings import BATCH_SIZE
from geo.data_reading.utils import loadImage
from geo.data_reading.augmentations import flipImage, cropMultiChannelImage, rotateImage 

def resizeImage(image, h, w):
    resized_image = np.zeros(shape=(33, h, w))
    for i in range(image.shape[0]):
        resized_image[i] = cv2.resize(image[i], (h, w), interpolation=cv2.INTER_LINEAR)

    return resized_image

def getDatasetChunk(samples):
    for i in range(0, len(samples), BATCH_SIZE):
        yield samples[i:i+BATCH_SIZE]

def getNextImageBatch(samples, species_map, augment=False, fromTestSet=False):
    for chunk in getDatasetChunk(samples):
        x_batch = np.zeros((BATCH_SIZE, 33, 64, 64), dtype=np.uint8)
        y_batch = np.zeros((BATCH_SIZE, len(species_map.keys())))
        species_ids_batch = np.zeros(BATCH_SIZE)
        glc_ids_batch = np.zeros(BATCH_SIZE)
        current_batch_slot = 0

        for sample in chunk:
            x = loadImage(sample, fromTestSet)
            if(augment):
                if np.random.random_sample() > 0.5:
                    x = flipImage(x)
                if np.random.random_sample() > 0.5:
                    x = rotateImage(x, 90)
                if np.random.random_sample() > 0.5:
                    x = cropMultiChannelImage(x)
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
        
def nextTestBatch(samples, species_map):
    for x, _, _, glc_ids in getNextImageBatch(samples, species_map, fromTestSet=True):
        yield (x, glc_ids)
