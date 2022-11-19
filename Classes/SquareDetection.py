import numpy as np
import cv2


class SquareDetection:
    def __init__(self, centroid: tuple[int], points: list[tuple[int]], distance: int, img=None):
        self.centroid = centroid
        self.points = points
        self.distance = distance
        self.img = img

        # Showing img or not
        self.draw_pixels()
        self.draw_centroid()
        self.show_img()


    def get_cropped_img(self, img: np.array):
        top = int(abs(self.centroid[0] - self.distance)), int(self.centroid[1] - self.distance)
        bot = int(self.centroid[0] + self.distance), int(abs(self.centroid[1] + self.distance))
        h, w = bot[1] - top[1], bot[0] - top[0]
        return [img[top[1]:top[1] + h, top[0]:top[0] + w], top, bot]

    def get_if_match(self):
        print(len(self.points))
        return True

    def draw_pixels(self):
        for p in self.points:
            self.img = cv2.circle(self.img, (p[0], p[1]), radius=0, color=(100, 100, 255), thickness=4)

    def draw_centroid(self):
        self.img = cv2.circle(self.img, (self.centroid[0], self.centroid[1]), radius=0, color=(125, 0, 255),
                              thickness=5)

    def show_img(self):
        cv2.imshow("Result", self.img)
        cv2.waitKey(0)
