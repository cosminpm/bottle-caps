import numpy as np

from Classes.SquareDetection import SquareDetection
from aux_scripts import distance_between_two_points
from Classes.KPsDcps import SIFTMatch

MIN_NUM_POINTS_IN_SQUARE = 13
MAX_DISTANCE = 50


class Detection:
    def __init__(self, name_cap: str, sift_match: SIFTMatch, img=None):
        self.squares = {}
        self.img = img
        self.sift_match = sift_match
        self.name = name_cap
        self.detect_centroids()

    def detect_centroids(self) -> None:
        already_in_square = set()
        for origin_point in self.sift_match.lst_pix:
            # p_X is on the horizontal axis and p_Y is on the vertical axis
            pMaxX, pMinX, pMaxY, pMinY = origin_point, origin_point, origin_point, origin_point
            already_in_square.add(origin_point)

            points_list = []
            for p in self.sift_match.lst_pix:
                # Checks if the cap is in the range of the max caps
                if (distance_between_two_points(pMaxX, p) < MAX_DISTANCE
                    or distance_between_two_points(pMinX, p) < MAX_DISTANCE
                    or distance_between_two_points(pMaxY, p) < MAX_DISTANCE
                    or distance_between_two_points(pMinY, p) < MAX_DISTANCE) \
                        and p not in already_in_square:
                    already_in_square.add(p)
                    points_list.append(p)
                    # Check horizontally
                    if p[0] > pMaxX[0]:
                        pMaxX = p
                    elif p[0] < pMinX[0]:
                        pMinX = p
                    # Check vertically
                    if p[1] > pMaxY[1]:
                        pMaxY = p
                    elif p[1] < pMinY[1]:
                        pMinY = p

            if len(points_list) > MIN_NUM_POINTS_IN_SQUARE:
                square = SquareDetection(points_list, pMaxX=pMaxX, pMinX=pMinX, pMaxY=pMaxY, pMinY=pMinY,
                                         name=self.name, img=self.img)
                if square.centroid not in self.squares:
                    self.squares[square.centroid] = [square]
                else:
                    self.squares[square.centroid] = self.squares[square.centroid].append(square)

    def get_all_squares(self) -> list[SquareDetection]:
        squares = []
        for key in self.squares.keys():
            for square in self.squares[key]:
                squares.append(square)
        return squares

    def get_cropped_squares(self, img: np.array) -> list:
        croppeds = []
        for key in self.squares.keys():
            for square in self.squares[key]:
                croppeds.append(square.get_cropped_img(img))
        return croppeds

    def set_prng_match(self) -> None:
        for square in self.get_all_squares():
            square.set_prng_match(len(self.sift_match.sift_cap.kps))

    def get_max_matches(self) -> int:
        num_matches = 0
        for square in self.get_all_squares():
            num_matches = max(num_matches, len(square.points))
        return num_matches

    # Draw
    def draw_all_squares(self, img: np.array) -> np.ndarray:
        squares = self.get_all_squares()
        for sq in squares:
            img = sq.draw_square()
        return img

    def draw_percentage(self, img: np.ndarray) -> np.ndarray:
        squares = self.get_all_squares()
        for square in squares:
            img = square.draw_percentage()
        return img

    def draw_name(self, img: np.ndarray) -> np.ndarray:
        squares = self.get_all_squares()
        for square in squares:
            img = square.draw_name(self.name)
        return img
