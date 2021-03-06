"""
Utils
"""

import numpy as np
import matplotlib.pyplot as plt
from threading import Thread, Lock
from PIL import Image
from scipy import ndimage
import os

IMG_TYPE_LIST = ['jpg', 'png']
INTERPOLATION_ORDER = 1


def show(img):
    if len(img.shape) < 3:
        plt.imshow(img, 'gray')
    else:
        plt.imshow(img)
    plt.show()


def shift_image(im, shift, interpolation_order=INTERPOLATION_ORDER):
    fractional, integral = np.modf(shift)
    if fractional.any():
        order = interpolation_order
    else:  # Disable interpolation
        order = 0
    return ndimage.shift(im, shift, order=order)


def readimg(file):
    return np.array(Image.open(file)) / 255


def check_if_img(filename):
    """
    basic check that helps load only imgs from a folder.
    """
    pos = filename.find('.')
    if pos < 0:
        return False
    return filename[pos + 1:].lower() in IMG_TYPE_LIST


def _read_images_in_chunks(folder_name, chunk, image_list, lock):
    for i, image_name in chunk:
        image = readimg('/'.join([folder_name, image_name]))
        image_list[int(i)] = image


def load_folder(folder_name, num_threads=6):
    """
    load all imgs from a folder and sorts them according to their names.
    """
    file_names = [filename for filename in os.listdir(folder_name) if check_if_img(filename)]
    file_names.sort()
    sorted_images_with_indices = list(zip(range(len(file_names)), file_names))
    chunks = np.array_split(sorted_images_with_indices, num_threads)
    images = [None] * len(file_names)
    thread_lst = []
    lock = Lock()
    for chunk in chunks:
        t = Thread(target=_read_images_in_chunks, args=(folder_name, chunk, images, lock))
        t.start()
        thread_lst.append(t)
    for t in thread_lst:
        t.join()
    return np.array(images)


def img2cv2(img):
    """
    :param img: a float RGB image in the range [0,1]
    :return: BGR image in range [0, 255]
    """
    return np.flip(np.clip(img * 255, 0, 255).astype(np.uint8), axis=2)
