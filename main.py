import cv2
import numpy as np
import os

from aux_scripts import distance_between_two_points
from kp_and_descriptors import compare_two_imgs

DISTANCE = 50
MY_CAPS_IMGS_FOLDER = "./caps_imgs/"


def read_img(img_path: str):
    return cv2.cvtColor(cv2.imread(img_path), 1)


def detect_squares(points: set, max_distance: int):
    centroid_distance = []
    already_in_square = set()
    for origin_point in points:
        # p_X is on the horizontal axis and p_Y is on the vertical axis
        pMaxX, pMinX, pMaxY, pMinY = origin_point, origin_point, origin_point, origin_point
        already_in_square.add(origin_point)

        centroid_list = []
        for p in points:
            # Checks if the cap is in the range of the max caps
            if (distance_between_two_points(pMaxX, p) < max_distance
                or distance_between_two_points(pMinX, p) < max_distance
                or distance_between_two_points(pMaxY, p) < max_distance
                or distance_between_two_points(pMinY, p) < max_distance) \
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
        if len(centroid_list) > 5:
            pTopLeft = (pMinX[0], pMaxY[1])
            pBotRight = (pMaxX[0], pMinY[1])
            centroid = np.array(centroid_list).mean(axis=0)
            centroid = tuple([int(i) for i in centroid])
            dis = max(distance_between_two_points(centroid, pTopLeft), distance_between_two_points(centroid, pBotRight))
            centroid_distance.append([centroid, dis])
    return centroid_distance


def get_cropped_squares(squares: list, img: np.ndarray):
    croppeds = []
    for s in squares:
        top = (int(abs(s[0][0] - s[1])), int(s[0][1] - s[1]))
        bot = (int(s[0][0] + s[1]), int(abs(s[0][1] + s[1])))
        h, w = bot[1] - top[1], bot[0] - top[0]
        croppeds.append([img[top[1]:top[1] + h, top[0]:top[0] + w].copy(), top, bot])
    return croppeds


def look_in_all_images(photo_str: str):
    entries = os.listdir(MY_CAPS_IMGS_FOLDER)
    photo_img = read_img(photo_str)
    for img in entries:
        cap_str = MY_CAPS_IMGS_FOLDER + img
        cap_img = read_img(cap_str)
        photo_img = draw_squares_detection(cap_img, photo_img, img)

    cv2.imshow("Result", photo_img)
    cv2.waitKey(0)


def draw_squares_detection(cap_img, photo_img, name_cap):
    points = compare_two_imgs(img_cap=cap_img, img_photo=photo_img)
    squares = detect_squares(points, DISTANCE)

    # Check if we have at least one match
    if len(squares) > 0:
        cropped = get_cropped_squares(squares, photo_img)
        for crop in cropped:
            photo_img = cv2.rectangle(photo_img, crop[1], crop[2], (255, 100, 0), 3)
            photo_img = cv2.putText(photo_img, name_cap, crop[1], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0))
    return photo_img


if __name__ == '__main__':
    look_in_all_images("./test_images/3.jpg")
