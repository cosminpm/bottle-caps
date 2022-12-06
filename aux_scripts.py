import os

import cv2
import numpy as np
import sys

from Classes.KPsDcps import SIFTApplied


def find_dominant_color(img: np.ndarray) -> tuple[int, int, int]:
    colors = {}
    for pix in img[0]:
        pix = tuple(pix)
        pix = (pix[0] // 2, pix[1] // 2, pix[2] // 2)
        if pix not in colors:
            colors[pix] = 1
        else:
            colors[pix] += 1
    dominant = max(colors, key=colors.get)
    return tuple((int(dominant[0] * 2), int(dominant[1] * 2), int(dominant[2] * 2)))


def compare_if_same_color(c1: np.ndarray, c2: np.ndarray, ratio: float) -> bool:
    sum_numb_1 = c1[0] / (255 * 3) + c1[1] / (255 * 3) + c1[2] / (255 * 3)
    sum_numb_2 = c2[0] / (255 * 3) + c2[1] / (255 * 3) + c2[2] / (255 * 3)
    return abs(sum_numb_1 - sum_numb_2) > ratio


def distance_between_two_points(p1: tuple, p2: tuple) -> float:
    dist = ((abs(p1[0] - p2[0]) ** 2) + (abs(p1[1] - p2[1]) ** 2)) ** 0.5
    return dist


def read_img(img_path: str) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(img_path), 1)


def get_mid_point(p1: tuple[int], p2: tuple[int]) -> tuple[int, int]:
    return tuple((int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2)))


def rgb_to_bgr(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple((rgb[2], rgb[1], rgb[0]))


def get_name_from_path(path: str) -> str:
    return path.split("/")[-1]


def resize_img_pix_with_name(cap_path, path_output, pix):
    cap_name = get_name_from_path(cap_path)
    lst_name_cap = cap_name.split(".")
    cap_name = lst_name_cap[0] + "_{}".format(str(pix)) + "." + lst_name_cap[-1]
    output = resize_image(cap_path, pix, pix, path_output, cap_name)
    return output


def resize_image(path_to_image, width, height, where_save, name_output):
    src = read_img(path_to_image)
    resized = cv2.resize(src, (width, height))
    output = where_save + name_output
    cv2.imwrite(output, resized)
    return output


def resize_all_images(path, output, size):
    files = os.listdir(path)
    print(files)
    for file in files:
        resize_img_pix_with_name(path + file, output, size)


def get_kps_path(path):
    files = os.listdir(path)
    for file in files:
        img = read_img(path + file)
        print("File {} with {} kps".format(file, len(SIFTApplied(img).kps)))


if __name__ == '__main__':
    resize_all_images("./caps_imgs/", "./resized_caps_imgs/", 200)
    get_kps_path("./resized_caps_imgs/")
