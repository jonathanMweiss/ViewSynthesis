"""
We assume that the motion between cameras is fixed.

if we know the motion we know how much the cameras are distant from one to the other,
thus we know how much shift there should be to get the closest item

"""

from Utils import *
from threading import Thread, Lock
from scipy import ndimage


class ImFocus:
    def __init__(self, path, num_threads, interpolation_oreder=INTERPOLATION_ORDER):
        self.imgs = None
        self.num_images = 0
        self.num_threads = num_threads
        self.mutex = Lock()
        self.interpolation_order = interpolation_oreder
        self.load_new_img_set(path)

    def load_new_img_set(self, path):
        self.imgs = load_folder(path, self.num_threads)
        self.num_images = self.imgs.shape[0]

    def shift_and_insert_to_list(self, shift_array, indices, lst):
        temp_lst = []
        for i in indices:
            temp_lst.append(shift_image(self.imgs[i], shift_array[i]))
        self.mutex.acquire()
        lst.extend(temp_lst)
        self.mutex.release()

    def _parallel_shift(self, shift_val):
        """
        assumes translation between imgs is fixed,
        computes the relative shift between the pictures.
        parameters:
        imgs:  a tensor [x, width, high, 3] where x is number of images.
        shift_val: the amount of shift to apply for each image in relation to the others.
        number_of_threads: how much to devided the data set.
        """
        sub_val = (self.num_images // 2)
        shifts = (np.arange(self.num_images) - sub_val) * shift_val
        shift_array = np.zeros((self.num_images, 3))
        shift_array[:, 1] = shifts[:]

        img_lst = []
        thread_lst = []
        indices = np.arange(shift_array.shape[0])

        for indices in np.array_split(indices, shift_array.shape[0] // self.num_threads):
            t = Thread(target=ImFocus.shift_and_insert_to_list, args=(self, shift_array, indices, img_lst))
            t.start()
            thread_lst.append(t)

        for t in thread_lst:
            t.join()
        return img_lst

    def mean_focus(self, shift_val):
        """
        compute the mean shift between images.
        """
        mean_img_lst = self._parallel_shift(shift_val)
        return np.clip(np.sum(mean_img_lst, axis=0) / self.num_images, 0, 1)

    def median_focus(self, shift_val):
        """
        compute the median shift between images.
        """
        img_lst = self._parallel_shift(shift_val)
        return np.clip(np.median(img_lst, axis=0), 0, 1)


if __name__ == '__main__':
    path = 'Pebbles-Stanford-2'
    focus_object = ImFocus(path, 2)
    show(focus_object.median_focus(-5.5))
