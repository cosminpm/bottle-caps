import cv2
import numpy as np
import os

from Classes.Detection import Detection
from aux_scripts import read_img
from kp_and_descriptors import compare_two_imgs

MY_CAPS_IMGS_FOLDER = "./caps_imgs/"


def look_in_all_images(photo_str: str):
    entries = os.listdir(MY_CAPS_IMGS_FOLDER)
    photo_img = read_img(photo_str)
    for name_img in entries:
        cap_str = MY_CAPS_IMGS_FOLDER + name_img
        cap_img = read_img(cap_str)
        photo_img = draw_squares_detections(cap_img, photo_img, name_img)
    cv2.imshow("Result", photo_img)
    cv2.waitKey(0)


def get_all_detections(photo_str: str) -> list[Detection]:
    entries = os.listdir(MY_CAPS_IMGS_FOLDER)
    photo_img = read_img(photo_str)
    detections = []
    for name_img in entries:
        cap_str = MY_CAPS_IMGS_FOLDER + name_img
        cap_img = read_img(cap_str)
        detections.append(get_detection(cap_img, photo_img, name_img))
    return detections


def get_detection(cap_img: np.ndarray, photo_img: np.ndarray, name_cap) -> Detection or None:
    pix_kps = compare_two_imgs(img_cap=cap_img, img_photo=photo_img)
    detection = Detection(pix_kps, name_cap, photo_img)
    if len(detection.squares) > 0:
        return detection
    return None


def draw_squares_detections(cap_img: np.ndarray, photo_img: np.ndarray, name_cap):
    pix_kps = compare_two_imgs(img_cap=cap_img, img_photo=photo_img)
    detection = Detection(pix_kps, name_cap, photo_img)

    # Check if we have at least one match
    if len(detection.squares) > 0:
        squares = detection.get_all_squares()
        for square in squares:
            crop = square.get_cropped_img(photo_img)
            photo_img = square.draw_square(photo_img)
            photo_img = cv2.putText(photo_img, name_cap, crop[1], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0))
            photo_img = cv2.putText(photo_img, str(square.get_perc_match()), crop[2], cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (255, 255, 0))
    return photo_img


def get_value_const():
    pass
    # TODO: Get value of const depending on images


if __name__ == '__main__':
    look_in_all_images("./test_images/4.jpg")
