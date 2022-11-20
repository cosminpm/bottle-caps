import numpy as np

from Classes.SquareDetection import SquareDetection
from aux_scripts import distance_between_two_points

MIN_NUM_POINTS_IN_SQUARE = 10
MAX_DISTANCE = 50


class Detection:
    def __init__(self, pix_kps: set[tuple[int]], name_cap: str, img=None):
        self.pix_kps = pix_kps
        self.squares = {}
        self.img = img
        self.detect_centroids()
        self.name = name_cap

    def detect_centroids(self):

        already_in_square = set()
        for origin_point in self.pix_kps:
            # p_X is on the horizontal axis and p_Y is on the vertical axis
            pMaxX, pMinX, pMaxY, pMinY = origin_point, origin_point, origin_point, origin_point
            already_in_square.add(origin_point)

            points_list = []
            for p in self.pix_kps:
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
                square = SquareDetection(points_list, pMaxX=pMaxX, pMinX=pMinX, pMaxY=pMaxY, pMinY=pMinY, img=self.img)
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

    def get_cropped_squares(self, img: np.array):
        croppeds = []
        for key in self.squares.keys():
            for square in self.squares[key]:
                croppeds.append(square.get_cropped_img(img))
        return croppeds
