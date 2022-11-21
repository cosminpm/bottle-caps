import os

import numpy as np

from Classes.Detection import Detection
from aux_scripts import read_img
from Classes.KPsDcps import SIFTApplied, SIFTMatch

MY_CAPS_IMGS_FOLDER = "./caps_imgs/"


class DetectionManager:
    def __init__(self, photo_image):
        self.detections = {}
        self.photo_image = photo_image
        self.get_all_detections(self.photo_image)
        self.set_prng_match()

    def get_all_detections(self, photo_img: np.ndarray):
        entries = os.listdir(MY_CAPS_IMGS_FOLDER)
        for name_img in entries:
            cap_str = MY_CAPS_IMGS_FOLDER + name_img
            cap_img = read_img(cap_str)
            self.add_detection(cap_img, photo_img, name_img)

    def set_prng_match(self):
        for key in self.detections:
            self.detections[key].set_prng_match()

    def get_max_matches(self) -> int:
        max_matches = 0
        for key in self.detections:
            max_matches = max(max_matches, self.detections[key].get_max_matches())
        return max_matches

    def add_detection(self, cap_img: np.ndarray, photo_img: np.ndarray, name_cap) -> Detection or None:
        sift_cap = SIFTApplied(cap_img)
        sift_photo = SIFTApplied(photo_img)
        match = SIFTMatch(sift_cap, sift_photo)
        detection = Detection(name_cap, match, photo_img)
        if len(detection.squares) > 0:
            self.detections[name_cap] = detection

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
