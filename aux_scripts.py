import cv2
import numpy as np


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
