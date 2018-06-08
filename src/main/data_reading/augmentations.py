import module_support_main
import numpy as np
import data_paths_main as data_paths
import settings_main as stg
import cv2

def flipImage(image):
    return cv2.flip(image, 1)

def rotateImage(image, angle):
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), np.random.uniform(-angle, angle), 1.0)

    return cv2.warpAffine(image, M,(w, h))

def cropMultiChannelImage(image, top=0.1, left=0.1, bottom=0.1, right=0.1):

    h, w = image.shape[:2]

    t_crop = max(1, int(h * np.random.uniform(0, top)))
    l_crop = max(1, int(w * np.random.uniform(0, left)))
    b_crop = max(1, int(h * np.random.uniform(0, bottom)))
    r_crop = max(1, int(w * np.random.uniform(0, right)))

    image = image[:, t_crop:-b_crop, l_crop:-r_crop]    
    
    cropped_image = np.zeros(shape=(33, 64, 64))

    for i in range(image.shape[0]):
        cropped_image[i] = cv2.resize(image[i], (64, 64), interpolation=cv2.INTER_LINEAR)

    return cropped_image

def cropSingleChannelImage(image, top=0.1, left=0.1, bottom=0.1, right=0.1):

    h, w = image.shape[:2]

    t_crop = max(1, int(h * np.random.uniform(0, top)))
    l_crop = max(1, int(w * np.random.uniform(0, left)))
    b_crop = max(1, int(h * np.random.uniform(0, bottom)))
    r_crop = max(1, int(w * np.random.uniform(0, right)))

    image = image[t_crop:-b_crop, l_crop:-r_crop]    

    return cv2.resize(image, (64, 64), interpolation=cv2.INTER_LINEAR)