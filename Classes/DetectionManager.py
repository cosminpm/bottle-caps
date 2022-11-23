import os

import numpy as np

from Classes.SquareDetection import SquareDetection
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

    def get_all_squares(self) -> list[SquareDetection]:
        squares = []
        for key in self.detections:
            for square in self.detections[key].get_all_squares():
                squares.append(square)
        return squares

    # TODO: Matches can't be one to one, the method must look all overlaping sets and then take one of them that's why case 3 it's failing
    def detect_non_overlapping_squares(self):
        squares = self.get_all_squares()
        not_overlapping = set()
        squares_aux = squares.copy()

        for s1 in squares:
            n_overlaps = 0
            squares_aux.pop(0)
            for s2 in squares_aux:
                area_overlap = s1.is_overlap(s2)
                # If there is overlap
                if area_overlap > s1.area() or area_overlap > s2.area():
                    n_overlaps += 1
                    if s1.percentage_match > s2.percentage_match:
                        not_overlapping.add(s1)
                    else:
                        not_overlapping.add(s2)
            squares_aux.append(s1)
            if n_overlaps == 0:
                not_overlapping.add(s1)

        print("tamano overlap", len(not_overlapping))
        return not_overlapping

    # Draw
    def draw_squares_detections(self):
        squares = self.detect_non_overlapping_squares()
        for square in squares:
            self.photo_image = square.draw_square()

    def draw_percentage(self):
        for key in self.detections:
            self.photo_image = self.detections[key].draw_percentage(self.photo_image)

    def draw_name(self):
        for key in self.detections:
            self.photo_image = self.detections[key].draw_name(self.photo_image)
