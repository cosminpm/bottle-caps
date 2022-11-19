import numpy as np


class SquareDetection:
    def __init__(self, centroid: tuple[int], points: list[tuple[int]], distance: int):
        self.centroid = centroid
        self.points = points
        self.distance = distance

    def get_cropped_img(self, img: np.array):
        top = int(abs(self.centroid[0] - self.distance)), int(self.centroid[1] - self.distance)
        bot = int(self.centroid[0] + self.distance), int(abs(self.centroid[1] + self.distance))
        h, w = bot[1] - top[1], bot[0] - top[0]
        return [img[top[1]:top[1] + h, top[0]:top[0] + w], top, bot]

    def get_if_match(self):
        # TODO
        pass
