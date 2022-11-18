import numpy as np

from aux_scripts import distance_between_two_points

MIN_NUM_OF_CENTROIDS = 7
MAX_DISTANCE = 50


class Detection:
    def __init__(self, pix_kps: set[tuple[int]]):
        self.pix_kps = pix_kps
        self.squares = self.detect_squares()

    def detect_squares(self):
        squares = []
        already_in_square = set()
        for origin_point in self.pix_kps:
            # p_X is on the horizontal axis and p_Y is on the vertical axis
            pMaxX, pMinX, pMaxY, pMinY = origin_point, origin_point, origin_point, origin_point
            already_in_square.add(origin_point)

            centroid_list = []
            for p in self.pix_kps:
                # Checks if the cap is in the range of the max caps
                if (distance_between_two_points(pMaxX, p) < MAX_DISTANCE
                    or distance_between_two_points(pMinX, p) < MAX_DISTANCE
                    or distance_between_two_points(pMaxY, p) < MAX_DISTANCE
                    or distance_between_two_points(pMinY, p) < MAX_DISTANCE) \
                        and p not in already_in_square:
                    already_in_square.add(p)
                    centroid_list.append(p)
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
            if len(centroid_list) > MIN_NUM_OF_CENTROIDS:
                pTopLeft = (pMinX[0], pMaxY[1])
                pBotRight = (pMaxX[0], pMinY[1])
                centroid = np.array(centroid_list).mean(axis=0)
                centroid = tuple([int(i) for i in centroid])
                dis = max(distance_between_two_points(centroid, pTopLeft),
                          distance_between_two_points(centroid, pBotRight))
                squares.append([centroid, dis])
        return squares

    def get_cropped_squares(self, img: np.array):
        croppeds = []
        for s in self.squares:
            top = (int(abs(s[0][0] - s[1])), int(s[0][1] - s[1]))
            bot = (int(s[0][0] + s[1]), int(abs(s[0][1] + s[1])))
            h, w = bot[1] - top[1], bot[0] - top[0]

            croppeds.append([img[top[1]:top[1] + h, top[0]:top[0] + w].copy(), top, bot])
        return croppeds