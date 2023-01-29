import json
import os

import cv2
import numpy as np

from Scripts.blobs import get_avg_size_all_blobs
from Scripts.HTC import hough_transform_circle

DEBUG_BLOB = False
MY_CAPS_IMGS_FOLDER = r"caps-s3"
DATABASE_FODLER = r"caps_db-s3"


def read_img(img_path: str) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(img_path), 1)


def rgb_to_bgr(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple((rgb[2], rgb[1], rgb[0]))


def rgb_to_bgr(r: int, g: int, b: int) -> tuple[int, int, int]:
    return tuple((b, g, r))


def get_name_from_path(path: str) -> str:
    return path.split("/")[-1]


def resize_img_pix_with_name(cap_path, path_output, pix):
    cap_name = get_name_from_path(cap_path)
    lst_name_cap = cap_name.split(".")
    cap_name = lst_name_cap[0] + "_{}".format(str(pix)) + "." + lst_name_cap[-1]
    output = resize_image_and_save(cap_path, pix, pix, path_output, cap_name)
    return output


def resize_image(src, factor):
    height, width = src.shape[:2]
    return cv2.resize(src, (int(src * factor), int(height * factor)))


def resize_image_and_save(path_to_image, width, height, where_save, name_output):
    src = read_img(path_to_image)
    resized = cv2.resize(src, (width, height))
    output = where_save + name_output
    cv2.imwrite(output, resized)
    return output


def resize_all_images(path, output, size):
    files = os.listdir(path)
    for file in files:
        resize_img_pix_with_name(path + file, output, size)


def crate_db_for_cap(cap_name, folder: str):
    cap_path = os.path.join(folder, cap_name)

    cap_img = cv2.imread(cap_path)
    cap_img = cv2.cvtColor(cap_img, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()

    kps, dcps = sift.detectAndCompute(cap_img, None)

    keypoints_list = [[kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response, kp.octave, kp.class_id] for kp in kps]

    dcps = dcps.tolist()[:200]

    entry = {
        "name": cap_name,
        "path": cap_path,
        "kps": keypoints_list,
        "dcps": dcps
    }
    cap_name = cap_name.split(".")[0]
    cap_result = os.path.join(DATABASE_FODLER, cap_name)
    with open(cap_result + ".json", "w") as outfile:
        json.dump(entry, outfile)


def create_json_for_all_caps():
    entries = os.listdir(MY_CAPS_IMGS_FOLDER)
    for name_img in entries:
        crate_db_for_cap(name_img, MY_CAPS_IMGS_FOLDER)


if __name__ == '__main__':
    create_json_for_all_caps()
