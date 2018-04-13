 
from data_reading import imageList_generator

import data_paths

if __name__ == '__main__':
    imageList_generator.generate__train_image_list()
    x = np.load(data_paths.image_train_set_x)
    print(x)