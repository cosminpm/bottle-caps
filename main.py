import cv2
import numpy as np
import os

from Classes.Detection import Detection
from aux_scripts import read_img
from kp_and_descriptors import compare_two_imgs

MY_CAPS_IMGS_FOLDER = "./caps_imgs/"

def look_in_all_images(photo_str: str):
    photo_img = read_img(photo_str)
    detections = get_all_detections(photo_img)
    set_prng_match(detections)

    photo_img = draw_squares_detections(detections, photo_img)
    photo_img = draw_percentage(detections, photo_img)
    photo_img = draw_name(detections, photo_img)

    cv2.imshow("Result", photo_img)
    cv2.waitKey(0)


def get_all_detections(photo_img: np.ndarray) -> list[Detection]:
    entries = os.listdir(MY_CAPS_IMGS_FOLDER)
    detections = []
    for name_img in entries:
        cap_str = MY_CAPS_IMGS_FOLDER + name_img
        cap_img = read_img(cap_str)
        detection = get_detection(cap_img, photo_img, name_img)
        if detection:
            detections.append(detection)
    return detections


def remove_overlapping_detections():
    pass


def get_detection(cap_img: np.ndarray, photo_img: np.ndarray, name_cap) -> Detection or None:
    pix_kps = compare_two_imgs(img_cap=cap_img, img_photo=photo_img)
    detection = Detection(pix_kps, name_cap, photo_img)
    if len(detection.squares) > 0:
        return detection
    return None


def set_prng_match(detections: list[Detection]):
    max_matches = get_max_matches(detections)
    for detection in detections:
        detection.set_prng_match(max_matches + 1)


def get_max_matches(detections: list[Detection]) -> int:
    max_matches = 0
    for detection in detections:
        max_matches = max(max_matches, detection.get_max_matches())
    return max_matches


# Draw
def draw_squares_detections(detections: list[Detection], img: np.ndarray):
    for detection in detections:
        img = detection.draw_all_squares(img)
    return img


def draw_percentage(detections: list[Detection], img: np.ndarray):
    for detection in detections:
        img = detection.draw_percentage(img)
    return img


def draw_name(detections: list[Detection], img: np.ndarray):
    for detection in detections:
        img = detection.draw_name(img)
    return img


def get_value_const():
    pass
    # TODO: Get value of const depending on images


if __name__ == '__main__':
    look_in_all_images("./test_images/4.jpg")
