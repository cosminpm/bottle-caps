import cv2
import numpy as np
import os

from Classes.Detection import Detection
from Classes.DetectionManager import DetectionManager
from aux_scripts import read_img
from kp_and_descriptors import compare_two_imgs


def look_in_all_images(photo_str: str):
    photo_img = read_img(photo_str)
    det_manager = DetectionManager(photo_img)
    det_manager.set_prng_match()

    det_manager.draw_squares_detections()
    det_manager.draw_percentage()
    det_manager.draw_name()

    cv2.imshow("Result", det_manager.photo_image)
    cv2.waitKey(0)


def remove_overlapping_detections():
    pass


def get_value_const():
    pass
    # TODO: Get value of const depending on images


if __name__ == '__main__':
    look_in_all_images("./test_images/4.jpg")
