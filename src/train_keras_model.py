 
from data_reading import imageList_generator
from data_reading import batch_generator as bg
import data_paths


if __name__ == '__main__':
    x, y = bg.getNextImageBatch
