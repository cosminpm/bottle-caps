import json
import os

import cv2
import numpy as np

from Scripts.HTC import hough_transform_circle
from Scripts.blobs import get_avg_size_all_blobs
from aux_scripts import read_img, rgb_to_bgr

MY_CAPS_IMGS_FOLDER = r"C:\Users\cosmi\Desktop\BottleCaps\caps_db"
MATCHER = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
SIFT = cv2.SIFT_create()
MAX_MATCHES = 100
MIN_MATCH_NUMBER = 20


def get_rectangles(circles: list[int, int, int]):
    rectangles = []
    for x, y, r in circles:
        x1 = x - r
        y1 = y - r
        width = r * 2
        height = r * 2
        rectangles.append((x1, y1, width, height))
    return rectangles


# Return the json file with that is the best match for that file
def get_best_match(dcp_rectangle) -> dict or None:
    matches = compare_all_dcps(dcp_rectangle)
    cap_file = {'percentage': 0,
                'path_file': None}
    for match in matches:
        if len(match[0]) > cap_file['percentage']:
            cap_file['percentage'] = len(match[0])
            cap_file['path_file'] = match[1]
    return cap_file


# Returns a list of (matches, path_to_json) from the dbs that compared with the file
def compare_all_dcps(dcp_rectangle):
    entries = os.listdir(MY_CAPS_IMGS_FOLDER)
    matches = []
    for name_img in entries:
        cap_str = os.path.join(MY_CAPS_IMGS_FOLDER, name_img)
        kps_cap, dcps_cap = get_kps_and_dcps_from_json(cap_str)

        match = (compare_dcps(dcps_cap, dcp_rectangle), cap_str)
        matches.append(match)
    return matches


def get_name_from_json(path):
    with open(path, "r") as file:
        data = json.load(file)
        name = data["name"]
        name = name.split('.')[-2][:-4]
    return name


def get_kps_and_dcps_from_json(path):
    with open(path, "r") as file:
        data = json.load(file)
        keypoints = data["kps"]
        descriptors = np.array(data["dcps"])
        keypoints = [cv2.KeyPoint(x, y, _size, _angle, _response, _octave, _class_id) for
                     x, y, _size, _angle, _response, _octave, _class_id in keypoints]
    return keypoints, descriptors


def cropp_image_into_rectangles(photo_image: np.ndarray, rectangles: list):
    cropped_images = []
    for x, y, w, h in rectangles:
        cropped_image = photo_image[y:y + h, x:x + w]
        cropped_images.append((cropped_image, (x, y, w, h)))
    return cropped_images


def get_dcp_and_kps(img: np.ndarray):
    return SIFT.detectAndCompute(img, None)


def compare_two_images(photo: np.ndarray, cap: np.ndarray):
    cap_kps, cap_dcp = get_dcp_and_kps(cap)
    photo_kps, photo_dcp = get_dcp_and_kps(photo)
    matches = compare_dcps(cap_dcp, photo_dcp)
    return matches


def compare_dcps(cap_dcp, photo_dcp):
    if cap_dcp.dtype != photo_dcp.dtype:
        if cap_dcp.dtype == np.float32:
            photo_dcp = np.array(photo_dcp, dtype=np.float32)
        else:
            cap_dcp = np.array(cap_dcp, dtype=np.float32)
    matches = MATCHER.match(cap_dcp, photo_dcp)
    matches = sorted(matches, key=lambda x: x.distance)[:MAX_MATCHES]
    return matches


def get_dict_all_matches(path_to_image: str) -> list[dict]:
    img = cv2.imread(path_to_image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, avg_size = get_avg_size_all_blobs(img.copy())
    _, circles = hough_transform_circle(img, avg_size)

    # Get the positions of the rectangles
    rectangles = get_rectangles(circles)
    # Crop the images from the rectangles
    cropped_images = cropp_image_into_rectangles(img, rectangles)

    # Final dictionary which will contain all the positions and info from the cap
    caps_matches = []

    for rectangle_image, pos_rectangle in cropped_images:
        _, dcp_rectangle = get_dcp_and_kps(rectangle_image)

        # Get the best possible match for each cap
        best_match_json = get_best_match(dcp_rectangle)
        # Get the position of the rectangle
        best_match_json['positions'] = {"x": pos_rectangle[0],
                                        "y": pos_rectangle[1],
                                        "w": pos_rectangle[2],
                                        "h": pos_rectangle[3]}
        best_match_json['name'] = get_name_from_json(best_match_json['path_file'])
        caps_matches.append(best_match_json)

    return caps_matches


def filter_matches(all_caps_matches: list[dict]) -> (list[dict], list[dict]):
    good_matches = []
    bad_matches = []
    for match in all_caps_matches:
        if match['percentage'] > MIN_MATCH_NUMBER:
            good_matches.append(match)
        else:
            bad_matches.append(match)
    return good_matches, bad_matches


def draw_match(img, match, color_name, color_circle):
    match_pos = match['positions']
    x, y, w, h = match_pos['x'], match_pos['y'], match_pos['w'], match_pos['h']
    center = (x + int(w / 2), y + int(h / 2))
    radius = int(w / 2)
    name = match['name']
    cv2.circle(img, center, radius, color_circle, 4)
    cv2.putText(img, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_name, 1, cv2.LINE_AA)
    return img


def draw_matches(path_to_image: str):
    GREEN_NAME = rgb_to_bgr(124, 252, 0)
    GREEN_CIRCLE = rgb_to_bgr(50, 205, 50)

    RED_NAME = rgb_to_bgr(220, 20, 60)
    RED_CIRCLE = rgb_to_bgr(255, 0, 0)

    all_matches = get_dict_all_matches(path_to_image=path_to_image)

    good_matches, bad_matches = filter_matches(all_matches)

    # drawing good matches on image
    img = cv2.imread(path_to_image)
    for match in good_matches:
        draw_match(img, match, GREEN_NAME, GREEN_CIRCLE)

    for match in bad_matches:
        draw_match(img, match, RED_NAME, RED_CIRCLE)

    cv2.imshow("a", img)
    cv2.waitKey(0)


def main():
    path_to_image = r"..\photo_images\9.jpg"
    draw_matches(path_to_image=path_to_image)


if __name__ == '__main__':
    main()
