import tifffile
from geo.data_paths import patch_train, patch_test


def loadImage(sample, fromTestSet=False):
    patch_dir_name = sample[0]
    patch_id = sample[1]
    if fromTestSet:
        img = tifffile.imread(patch_test+'/{}/patch_{}.tif'.format(patch_dir_name, patch_id))
    else:
        img = tifffile.imread(patch_train+'/{}/patch_{}.tif'.format(patch_dir_name, patch_id))

    return img
