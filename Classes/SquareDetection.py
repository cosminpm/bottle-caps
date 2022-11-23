from __future__ import annotations

import numpy as np
import cv2

from aux_scripts import distance_between_two_points, get_mid_point, rgb_to_bgr
from Classes.KPsDcps import MAX_MATCHES

DEBUG = False
COLOR_PERCENTAGE = rgb_to_bgr((0, 0, 0))
THICKNESS_SQUARE = 2
COLOR_SQUARE = rgb_to_bgr((255, 43, 0))


class SquareDetection:
    def __init__(self, points: list[tuple[int]], pMaxX, pMinX, pMaxY, pMinY, name, img=None):
        self.points = points
        self.centroid = self.calc_centroid()
        self.distance = self.calc_distance(pMaxX=pMaxX, pMinX=pMinX, pMaxY=pMaxY, pMinY=pMinY)
        self.img = img
        self.name = name
        self.percentage_match = len(self.points) / MAX_MATCHES

        # Showing img or not
        self.debug()

    def calc_centroid(self):
        cen = np.array(self.points).mean(axis=0)
        tp = tuple([int(i) for i in cen])
        return tp

    def calc_distance(self, pMaxX, pMinX, pMaxY, pMinY):
        pTopLeft = (pMinX[0], pMaxY[1])
        pBotRight = (pMaxX[0], pMinY[1])
        return max(distance_between_two_points(self.centroid, pTopLeft),
                   distance_between_two_points(self.centroid, pBotRight))

    def get_cropped_img(self):
        top, bot = self.get_bot_and_top_corners()
        h, w = bot[1] - top[1], bot[0] - top[0]
        return [self.img[top[1]:top[1] + h, top[0]:top[0] + w], top, bot]

    def set_prng_match(self, max_detections: int):
        actual_detection = len(self.points) / max_detections
        self.percentage_match = max(actual_detection, self.percentage_match)

    # Drawing Methods
    def draw_pixels(self):
        for p in self.points:
            self.img = cv2.circle(self.img, (p[0], p[1]), radius=0, color=(100, 100, 255), thickness=4)

    def draw_centroid(self):
        self.img = cv2.circle(self.img, (self.centroid[0], self.centroid[1]), radius=0, color=(125, 0, 255),
                              thickness=5)

    def get_bot_and_top_corners(self):
        top = int(abs(self.centroid[0] - self.distance)), int(self.centroid[1] - self.distance)
        bot = int(self.centroid[0] + self.distance), int(abs(self.centroid[1] + self.distance))

        return top, bot

    def draw_square(self):
        crop = self.get_cropped_img()
        photo_img = cv2.rectangle(self.img, crop[1], crop[2], COLOR_SQUARE, THICKNESS_SQUARE, -1)
        return photo_img

    def draw_percentage(self):
        crop = self.get_cropped_img()
        mid_point = get_mid_point(crop[1], crop[2])
        percentage = "{:.2f}".format(self.percentage_match)
        photo_img = cv2.putText(self.img, percentage, mid_point,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_PERCENTAGE, 1, cv2.LINE_AA)
        return photo_img

    def draw_name(self):
        crop = self.get_cropped_img()
        photo_img = cv2.putText(self.img, self.name, crop[1],
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_PERCENTAGE, 1, cv2.LINE_AA)
        return photo_img

    def is_overlap(self, aux_square: SquareDetection):
        a = self.get_bot_and_top_corners()
        b = aux_square.get_bot_and_top_corners()
        dx = min(a[1][0], b[1][0]) - max(a[0][0], b[0][0])
        dy = min(a[1][1], b[1][1]) - max(a[0][1], b[0][1])
        if (dx >= 0) and (dy >= 0):
            return dx * dy
        else:
            return 0

    def area(self):
        return self.distance * 2 * self.distance

    # Show
    def show_img(self):
        cv2.imshow("Result", self.img)
        cv2.waitKey(0)

    def debug(self):
        if DEBUG:
            self.draw_pixels()
            self.draw_centroid()
            self.show_img()
