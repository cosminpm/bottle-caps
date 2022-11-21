import os

import numpy as np

from Classes.Detection import Detection
from aux_scripts import read_img
from kp_and_descriptors import compare_two_imgs

MY_CAPS_IMGS_FOLDER = "./caps_imgs/"


def get_detection(cap_img: np.ndarray, photo_img: np.ndarray, name_cap) -> Detection or None:
    pix_kps = compare_two_imgs(img_cap=cap_img, img_photo=photo_img)
    detection = Detection(pix_kps, name_cap, photo_img)
    if len(detection.squares) > 0:
        return detection
    return None


class DetectionManager:
    def __init__(self, photo_image):
        self.detections = {}
        self.photo_image = photo_image
        self.get_all_detections(self.photo_image)

    def get_all_detections(self, photo_img: np.ndarray):
        entries = os.listdir(MY_CAPS_IMGS_FOLDER)
        for name_img in entries:
            cap_str = MY_CAPS_IMGS_FOLDER + name_img
            cap_img = read_img(cap_str)
            detection = get_detection(cap_img, photo_img, name_img)
            if detection:
                self.detections[cap_str] = detection

    def set_prng_match(self):
        max_matches = self.get_max_matches()
        for key in self.detections:
            self.detections[key].set_prng_match(max_matches + 1)

    def get_max_matches(self) -> int:
        max_matches = 0
        for key in self.detections:
            max_matches = max(max_matches, self.detections[key].get_max_matches())
        return max_matches

    # Draw
    def draw_squares_detections(self):
        for key in self.detections:
            self.photo_image = self.detections[key].draw_all_squares(self.photo_image)

    def draw_percentage(self):
        for key in self.detections:
            self.photo_image = self.detections[key].draw_percentage(self.photo_image)

    def draw_name(self):
        for key in self.detections:
            self.photo_image = self.detections[key].draw_name(self.photo_image)
