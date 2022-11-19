import math
import numpy as np
import cv2

from aux_scripts import distance_between_two_points

DEBUG = False

class SquareDetection:
    def __init__(self, points: list[tuple[int]], pMaxX, pMinX, pMaxY, pMinY, img=None):
        self.points = points
        self.centroid = self.calc_centroid()
        self.distance = self.calc_distance(pMaxX=pMaxX, pMinX=pMinX, pMaxY=pMaxY, pMinY=pMinY)
        self.img = img

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

    def get_cropped_img(self, img: np.array):
        top = int(abs(self.centroid[0] - self.distance)), int(self.centroid[1] - self.distance)
        bot = int(self.centroid[0] + self.distance), int(abs(self.centroid[1] + self.distance))
        h, w = bot[1] - top[1], bot[0] - top[0]
        return [img[top[1]:top[1] + h, top[0]:top[0] + w], top, bot]

    # Drawing Methods
    def draw_pixels(self):
        for p in self.points:
            self.img = cv2.circle(self.img, (p[0], p[1]), radius=0, color=(100, 100, 255), thickness=4)

    def draw_centroid(self):
        self.img = cv2.circle(self.img, (self.centroid[0], self.centroid[1]), radius=0, color=(125, 0, 255),
                              thickness=5)
    # Show
    def show_img(self):
        cv2.imshow("Result", self.img)
        cv2.waitKey(0)

    def debug(self):
        if DEBUG:
            self.draw_pixels()
            self.draw_centroid()
            self.show_img()
