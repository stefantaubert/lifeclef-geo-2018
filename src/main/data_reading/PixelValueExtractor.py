import numpy as np
from tqdm import tqdm

def get_pixel_value(img, c_pixel):
    '''c_pixel: 1 -> 4 mittlersten Pixel; 2-> 16 innersten pixel; 3 -> 36 innersten pixel
        Berechnung: c_pixel^2 * 4 oder (2 * c_pixel) ^ 2        
        bei 1 -> 4 = (64-2*1) / 2 = 31 for i, j = 2*1
        bei 2 -> 16 = (64-2*2) / 2 = 30 for i, j = 2*2
        bei 3 -> 36 = (64-2*3) / 2 = 29 for i, j = 2*3
    '''
    img_dim = len(img)

    assert c_pixel > 0
    assert c_pixel <= img_dim / 2
    assert img_dim % 2 == 0
    assert img.shape == (img_dim, img_dim)

    if c_pixel == img_dim / 2:
        img_mean = img.mean()
        return img_mean

    dimension = c_pixel * 2
    values = []
    base_index = int((img_dim - (2*c_pixel)) / 2)

    for i in range(dimension):
        for j in range(dimension):
            values.append(img[base_index + i][base_index + j])

    assert len(values) == c_pixel * c_pixel * 4

    values_array = np.asarray(values)
    mean = values_array.mean()

    return mean