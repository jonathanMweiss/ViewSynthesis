from Utils import shift_image, load_folder

import numpy as np
from skimage.feature import register_translation
import matplotlib.pyplot as plt


def xslits_crop(shifts, alpha, start):
    """ To make an Xslit image we need to compute the centers to take strips from and the bandwidths of each
    center.
    we use the linear line created over the image space-time graph.
    t(s) = a * s + b ( we know thetranslation, that is t(s) actual value, a and b,
    and the centers are s of this equation)
    the bandwidths are the distance from one center to the middle between the other center.
    """
    cumulative_shifts = np.cumsum(shifts[beta:, 1])
    centers = (cumulative_shifts / alpha) + start
    centers = np.hstack((start, centers))  # function expects tuple.

    # pad with zero at begining and end to get mean between two centers
    bandwidths = ((np.pad(centers, (0, 1), 'constant', constant_values=0) -
                   np.pad(centers, (1, 0), 'constant', constant_values=0)) / 2)[1:-1]
    return centers, bandwidths


def make_xslit(images, centers, bandwidths, shifts):
    """
    we take the crops, the centers and the badnwidths, the shifts and we crop from each image the needed part.
    return an xslit image.
    """
    stitched_img = images[0, :,
                   centers[0].astype(np.int):np.floor(centers[0] + bandwidths[0] + shifts[0][1] / 2).astype(np.int)]

    for i in range(1, images.shape[0] - 1):
        if centers[i] >= 0:
            left_crop = np.ceil(centers[i] - bandwidths[i - 1] - shifts[i - 1][1] / 2).astype(np.int)
            right_crop = np.floor(centers[i] + bandwidths[i] + shifts[i][1] / 2).astype(np.int)
            stitched_img = np.hstack((stitched_img, images[i, :, left_crop:right_crop]))
        else:  # TODO need to check th values of a decending order.
            # centers are in decending order, no need to continue loop.
            return stitched_img
    i = images.shape[0] - 1
    return np.hstack((stitched_img, images[i, :,
                                    np.ceil(centers[i] - bandwidths[i - 1] - shifts[i - 1][1] / 2).astype(np.int):
                                    np.ceil(centers[i]).astype(np.int)]))


def make_panorama(images, shifts, center_pos):
    if center_pos + np.ceil(np.max(shifts, axis=0))[1] > images.shape[2]:
        right_crop = np.repeat(center_pos, images.shape[0]).astype(np.int)
        left_crop = center_pos - np.floor(shifts[:, 1]).astype(np.int)
    else:
        right_crop = center_pos + np.ceil(shifts[:, 1]).astype(np.int)
        left_crop = np.repeat(center_pos, images.shape[0]).astype(np.int)
    right_crop = right_crop.astype(np.int)
    left_crop = left_crop.astype(np.int)
    panorama = images[0, :, left_crop[0]: right_crop[0]]
    for i in range(1, images.shape[0] - 1):
        panorama = np.hstack((panorama, images[i, :, left_crop[i]: right_crop[i]]))
    return panorama


def render_image(image_set, shifts, beta, alpha, start, end):
    """
    Makes panorama from every image aviliable.
    Makes Xslit according to alpha and beta given.
    """
    if beta > image_set.shape[0] or beta < 0:
        msg = "Error! Beta value is: " + str(beta)
        raise Exception(msg)
    if start == end or abs(alpha) > 350:  # high alpha is almost panorama.
        return make_panorama(image_set, shifts, int(start))
    else:
        beta = int(beta)
        images = image_set[beta:]
        # I wanted to remove division by zero, this is the wanted result too.
        if alpha == 0:
            return images[0]
        new_shifts = shifts[beta:]
        if alpha < 0:
            start = end
        centers, bandwidths = xslits_crop(new_shifts, alpha, start)

        return make_xslit(images, centers, bandwidths, new_shifts)


if __name__ == '__main__':

    path_folder = 'train-in-snow'

    images = load_folder(path_folder)  # np.array([readimg(path_folder + "/" + img_name) for img_name in ret_val])

    # computing shifts for all images:
    imgNum = images.shape[0]
    shifts = np.zeros([imgNum - 1, 2])

    for i in range(imgNum - 1):
        shifts[i], _, _ = register_translation(images[i, :, :, 0], images[i + 1, :, :, 0], 100)
        correction = np.array([shifts[i][0], 0, 0])

    # making small corrections to images.
    for i in range(imgNum - 1):
        correction = np.array([shifts[i][0], 0, 0])
        images[i + 1] = shift_image(images[i + 1], correction)

    translation = np.sum(shifts, axis=0)[1]
    width = images[0].shape[1]
    beta = 0
    alpha = 3
    start, end = 0, width - 1

    a = render_image(images, shifts, beta, alpha, start, end)
    plt.imshow(a)
    plt.show()
    print("done")