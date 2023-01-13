import os

import cv2
import numpy as np

from Scripts.HTC import hough_transform_circle
from Scripts.blobs import get_avg_size_all_blobs
from aux_scripts import read_img

MY_CAPS_IMGS_FOLDER = r"C:\Users\cosmi\Desktop\BottleCaps\resized_caps_imgs\\"
MATCHER = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
SIFT = cv2.SIFT_create()
MAX_MATCHES = 100
MIN_MATCH_NUMBER = 13


def get_rectangles(circles: list[int, int, int]):
    rectangles = []
    for x, y, r in circles:
        x1 = x - r
        y1 = y - r
        width = r * 2
        height = r * 2
        rectangles.append((x1, y1, width, height))
    return rectangles


def get_all_detections(cropped_image):
    entries = os.listdir(MY_CAPS_IMGS_FOLDER)
    matches = []
    for name_img in entries:
        cap_str = MY_CAPS_IMGS_FOLDER + name_img
        cap_img = read_img(cap_str)
        match = (compare_two_images(cropped_image, cap_img), name_img)
        if len(match[0]) > MIN_MATCH_NUMBER:
            matches.append(match)

    greater = 0
    name = ""
    for match in matches:
        if len(match[0]) > greater:
            greater = len(match[0])
            name = match[1]

    cropped_image = cv2.putText(cropped_image, name, (10, 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return cropped_image


def cropp_image_into_rectangles(photo_image: np.ndarray, rectangles: list):
    cropped_images = []
    for x, y, w, h in rectangles:
        cropped_image = photo_image[y:y + h, x:x + w]
        cropped_images.append((cropped_image, (x, y, w, h)))
    return cropped_images


def compare_two_images(photo: np.ndarray, cap: np.ndarray):
    cap_kps, cap_dcp = SIFT.detectAndCompute(cap, None)
    photo_kps, photo_dcp = SIFT.detectAndCompute(photo, None)

    matches = MATCHER.match(cap_dcp, photo_dcp)
    matches = sorted(matches, key=lambda x: x.distance)[:MAX_MATCHES]
    return matches


def main():
    path_to_image = r"C:\Users\cosmi\Desktop\BottleCaps\photo_images\9.jpg"
    img = cv2.imread(path_to_image, 0)
    _, avg_size = get_avg_size_all_blobs(img.copy())
    _, circles = hough_transform_circle(img, avg_size)
    rectangles = get_rectangles(circles)
    cropped_images = cropp_image_into_rectangles(img, rectangles)

    for crop, tupla in cropped_images:
        x, y, w, h = tupla[0], tupla[1], tupla[2], tupla[3]
        img[y:y + h, x:x + w] = get_all_detections(crop)

    cv2.imshow("a", img)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
