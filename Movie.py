import cv2
import argparse
import numpy as np
from Utils import load_folder, img2cv2
import ImXslit as xslit
import os


# Construct the argument parser and parse the arguments
# TODO bug : colour seems off in the image.


def main():
    dir_path, output = 'train-in-snow', 'outputcheck.mp4'

    print('== loading image set ==')
    imgs_to_render = load_folder(dir_path, 6)

    print('== collecting shifts ==')
    imgs_to_render, shifts = xslit.get_shifts_and_corrected_imgs(imgs_to_render)
    images = []

    print('== Computing panoramas ==')
    _, width, _ = imgs_to_render[0].shape
    for i in range(width):
        rendered_img = xslit.compute(imgs_to_render, shifts, 0, 1000, i, i)
        images.append(img2cv2(rendered_img))

    # Determine the width and height from the first image
    frame = images[0]
    height, width, _ = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lower case
    out = cv2.VideoWriter(output, fourcc, 20.0, (width, height))

    print('== choosing images ==')
    for frame in images:
        out.write(frame)  # Write out frame to video

    # Release everything if job is finished
    print('== creating video ==')
    out.release()
    cv2.destroyAllWindows()

    print(f"== The output video name: {output} ==")


if __name__ == '__main__':
    main()
