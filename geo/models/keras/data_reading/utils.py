import module_support_main
import data_paths_main as data_paths
import tifffile

def loadImage(sample):
    patch_dir_name = sample[0]
    patch_id = sample[1]
    img = tifffile.imread(data_paths.patch_train+'/{}/patch_{}.tif'.format(patch_dir_name, patch_id))

    return img
