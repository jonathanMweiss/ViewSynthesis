"""
Utils
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

IMG_TYPE_LIST = ['jpg', 'png']
INTERPOLATION_ORDER = 1


def show(img):
    if len(img.shape) < 3:
        plt.imshow(img,'gray')
    else:
        plt.imshow(img)
    plt.show()


def readimg(file):
    return np.array(Image.open(file)) / 255


def check_if_img(filename):
    """
    basic check that helps load only imgs from a folder.
    """
    pos = filename.find('.')
    if pos < 0:
        return False
    return True if filename[pos + 1:].lower() in IMG_TYPE_LIST else False


def load_folder(folder_name):
    """
    load all imgs from a file and sorts them according to their names.
    """
    ret_val = [filename for filename in os.listdir(folder_name) if check_if_img(filename)]
    ret_val.sort()
    return np.array([readimg(folder_name + "/" + img_name) for img_name in ret_val])


def convert_to_pil(arr):
    return Image.fromarray((255 * np.clip(arr, 0 , 1)).astype('uint8'))


def im2Pil(img):
    pass # TODO.