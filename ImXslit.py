import numpy as np
import matplotlib.pyplot as plt

from skimage import data
from skimage.feature import register_translation
from skimage.feature.register_translation import _upsampled_dft


from Utils import *
import os
from skimage.feature import register_translation

class ImXslit:
    def __init__(self, controller):
        self.controller = controller #?
        self.image_container = None #?
        self.container = None #?
        self.imgToShow = None #?
        self.shifts = None #?
        if hasattr(self.controller, 'images'):
            self.computeShiftsNCorrect()

    def setContainer(self, container):
        self.container = container

    def getWidth(self):
        return self.controller.images[0].shape[1]

    def getNumFrame(self):
        return self.controller.images.shape[0]

    def render(self, alpha, beta, betaStop, start, end):
        image = self.renderImage(alpha, beta,betaStop, start, end)
        if 0 not in image.shape: # if there is no 'zero image' compute.
            self.imgToShow = ImageTk.PhotoImage(im2Pil(image))
        self.container.configure(image=self.imgToShow)

    def computeShiftsNCorrect(self):
        imgNum = self.controller.images.shape[0]
        self.shifts = np.zeros([imgNum - 1, 2])

        for i in range(imgNum - 1):
            self.shifts[i], _, _ = register_translation(
                self.controller.images[i, :, :, 0],
                self.controller.images[i + 1, :, :, 0],
                100)

        # making small corrections to images.
        for i in range(imgNum - 1):
            correction = np.array([self.shifts[i][0], 0, 0])
            self.controller.images[i + 1] = shift_image(self.controller.images[i + 1], correction)

    def xslitsCrop(self, shifts, alpha, start, end):
        cumShifts = np.cumsum(shifts[:, 1])
        translation = np.sum(shifts, axis=0)[1]
        centers = (cumShifts / alpha) + start
        centers = np.hstack((start, centers))
        # pad with zero at begin and end to get mean between two centers
        if alpha >= 0:
            bandwidths = ((np.pad(centers, (0, 1), 'constant', constant_values=0) -
                           np.pad(centers, (1, 0), 'constant', constant_values=0)) / 2)[1:-1]
        else:
            bandwidths = ((np.pad(centers, (1, 0), 'constant', constant_values=0) -
                           np.pad(centers, (0, 1), 'constant', constant_values=0)) / 2)[1:-1]
        return centers, bandwidths

    def makeXslit(self, images, centers, bandwidths, shifts):
        stitched_img = images[0, :, centers[0].astype(np.int):
                                    np.floor(centers[0] + bandwidths[0] + shifts[0][1] / 2).astype(np.int)]
        for i in range(1, images.shape[0] - 1):
            if centers[i] >= 0:
                left_crop = np.ceil(centers[i] - bandwidths[i - 1] - shifts[i - 1][1] / 2).astype(np.int)
                right_crop = np.floor(centers[i] + bandwidths[i] + shifts[i][1] / 2).astype(np.int)
                stitched_img = np.hstack((stitched_img, images[i, :, left_crop:right_crop]))
            else:
                # centers are in decending order, no need to continue loop.
                return stitched_img
        i = images.shape[0] - 1
        return np.hstack((stitched_img, images[i, :,
                                        np.ceil(centers[i] - bandwidths[i - 1] - shifts[i - 1][1] / 2).astype(np.int):
                                              np.ceil(centers[i]).astype(np.int)]))

    def makePanorama(self, images, shifts, centerPos):
        if centerPos + np.ceil(np.max(shifts, axis=0))[1] > images.shape[2]:
            right_crop = np.repeat(centerPos, images.shape[0]).astype(np.int)
            left_crop = centerPos - np.floor(shifts[:, 1]).astype(np.int)
        else:
            right_crop = centerPos + np.ceil(shifts[:, 1]).astype(np.int)
            left_crop = np.repeat(centerPos, images.shape[0]).astype(np.int)
        right_crop = right_crop.astype(np.int)
        left_crop = left_crop.astype(np.int)
        panorama = images[0, :, left_crop[0]: right_crop[0]]
        for i in range(1, images.shape[0] - 1):
            panorama = np.hstack((panorama, images[i, :, left_crop[i]: right_crop[i]]))
        return panorama

    def renderImage(self, alpha, beta, betaStop, start, end):
        """
        Makes panorama from every image aviliable.
        Makes Xslit according to alpha and beta given.
        """
        if beta > self.controller.images.shape[0] or beta < 0:
            msg = "Error! Beta value is: " + str(beta)
            raise Exception(msg)
        if start == end or abs(alpha) > 350:  # high alpha is almost panorama.
            return self.makePanorama(self.controller.images, self.shifts, int(start))
        else:
            beta = int(beta)
            betaStop = int(betaStop)
            images = self.controller.images[beta:betaStop]
            # I wanted to remove division by zero, this is the wanted result too.
            if alpha == 0:
                return self.controller.images[beta]
            new_shifts = self.shifts[beta: betaStop]
            # TODO ask Lirane if this should be changed by the gui??
            if alpha < 0:
                start ,end = end, start
            centers, bandwidths = self.xslitsCrop(new_shifts, alpha, start, end)

            return self.makeXslit(images, centers, bandwidths, new_shifts)

#
# # path_folder = './data/train-in-snow'
# # path_folder = '/Users/yonatanweiss/Downloads/Banana'
# path_folder = '/Users/yonatanweiss/PycharmProjects/lightfield/drive-download-20180627T063135Z-001/train-in-snow'
#
# # should be using the loading folder function:
# ret_val = [filename for filename in os.listdir(path_folder) if check_if_img(filename)]
# ret_val.sort()
# images = np.array([readimg(path_folder + "/" + img_name) for img_name in ret_val])
# # images = images[0:2]
#
# # computing shifts for all images:
# imgNum = images.shape[0]
# shifts = np.zeros([imgNum - 1, 2])
#
#
# for i in range(imgNum - 1):
#     shifts[i], _, _ = register_translation(images[i, :, :, 0], images[i + 1, :, :, 0], 100)
#     correction = np.array([shifts[i][0], 0, 0])
#
# # making small corrections to images.
# for i in range(imgNum - 1):
#     correction = np.array([shifts[i][0], 0, 0])
#     images[i + 1] = shift_image(images[i + 1], correction)
#
# translation = np.sum(shifts, axis=0)[1]
# width = images[0].shape[1]
# beta = 0
# alpha = 3
# start, end = 0, width - 1
#
# a = renderImage(images, shifts, beta, alpha ,start, end)
# plt.imshow(a)
# plt.show()
# print("done")