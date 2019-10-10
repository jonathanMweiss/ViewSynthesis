import cv2
import argparse
import numpy as np
from Utils import load_folder, img2cv2
import ImXslit as xslit
import os


# Construct the argument parser and parse the arguments
# TODO bug : colour seems off in the image.

def print_dict(dictionary):
    max_length = len(max(dictionary, key=len))
    for key in dictionary:
        print(f"{key}:{' ' * (max_length - len(key))} {dictionary[key]}")


def get_params():
    ap = argparse.ArgumentParser()
    ap.add_argument('-ext', '--extension', required=False, default='jpg', help='extension name. '
                                                                               'default is "jpg".')
    ap.add_argument('-o', '--output', required=False, default='outputcheck.mp4',
                    help='output video file')

    args = vars(ap.parse_args())
    print('== INPUTS ==')
    print_dict(args)

    # Arguments
    dir_path = 'train-in-snow'
    ext = args['extension']
    output = args['output']
    return dir_path, ext, output


def main():
    dir_path, ext, output = get_params()

    imgs_to_render = load_folder(dir_path, 6)

    print('collecting shifts')
    imgs_to_render, shifts = xslit.get_shifts_and_corrected_imgs(imgs_to_render)
    images = []

    print('Computing panoramas')
    _, width, _ = imgs_to_render[0].shape
    for i in range(width):
        rendered_img = xslit.compute(imgs_to_render, shifts, 0, 1000, i, i)
        images.append(img2cv2(rendered_img))

    # Determine the width and height from the first image
    # image_path = os.path.join(dir_path, images[0])
    frame = images[0]
    cv2.imshow('video', frame)
    height, width, channels = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lower case
    out = cv2.VideoWriter(output, fourcc, 20.0, (width, height))

    print('choosing images')
    for frame in images:
        out.write(frame)  # Write out frame to video
        cv2.imshow('video', frame)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):  # Hit `q` to exit
            break

    # Release everything if job is finished
    print('creating video')
    out.release()
    cv2.destroyAllWindows()

    print(f"The output video is {output}")


if __name__ == '__main__':
    main()