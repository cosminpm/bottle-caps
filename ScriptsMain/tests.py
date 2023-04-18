import json
import os

import numpy as np

from ScriptsMain.SIFT import get_dict_all_matches, detect_caps
from ScriptsMain.utils import read_img
import cv2


def get_all_names_from_all_matches(all_matches: list[dict]):
    result = set()
    for match in all_matches:
        result.add(match['name'])
    return result


def test_get_dict_all_matches_i_have():
    folder_photos = '../database/test-images/test-i-have'
    entries = os.listdir(folder_photos)

    file_solution = open('../database/test-images/solution-test-i-have.json')
    json_solution = json.load(file_solution)

    for entry in entries:
        path_to_image = os.path.join(folder_photos, entry)
        all_matches, _ = get_dict_all_matches(path_to_image)

        result_all_matches = get_all_names_from_all_matches(all_matches)
        expected_result = set(json_solution[entry])

        common_elements = result_all_matches.intersection(expected_result)

        print(f"For test {entry}, I got an accuracy of {len(common_elements) / len(expected_result)}")


def read_hsv(path: str):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2HSV)


def get_avg_hsv(hsv_img):
    height, width = hsv_img.shape[:2]
    center = (width // 2, height // 2)
    radius = min(center[0], center[1])

    # Create a circular mask
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    hsv_circle = cv2.bitwise_and(hsv_img, hsv_img, mask=mask)

    # Calculate the average HSV values over the circular region
    avg = cv2.mean(hsv_circle, mask=mask)[:3]

    # Convert the average HSV values to integers
    avg = [int(x) for x in avg]
    return avg


def get_hsv_mean_from_path(path: str):
    hsv_image = read_hsv(path)
    avg = get_avg_hsv(hsv_image)

    # Convert the average HSV values to BGR color space for display
    bgr_color = cv2.cvtColor(np.array([[avg]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0][0]

    # Create an image of the color using NumPy
    color_image = np.zeros((100, 100, 3), dtype=np.uint8)
    color_image[:, :] = bgr_color

    # Display the color image using OpenCV
    cv2.imshow('HSV Color', color_image)
    cv2.waitKey(0)

    print("Average HSV values:", avg)


def get_avg_hsv_from_picture(path_picture):
    img = read_img(path_picture)
    list_caps = detect_caps(img)
    for cap in list_caps:
        cap_array = cap[0]  # extract the NumPy array from the tuple
        hsv_image = cv2.cvtColor(cap_array.astype(np.uint8), cv2.COLOR_BGR2HSV)
        avg = get_avg_hsv(hsv_image)
        print(avg)


def get_current_accuracy():
    test_get_dict_all_matches_i_have()



def read_lab(path: str):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2LAB)

def get_avg_lab(lab_img):
    height, width = lab_img.shape[:2]
    center = (width // 2, height // 2)
    radius = min(center[0], center[1])

    # Create a circular mask
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    lab_circle = cv2.bitwise_and(lab_img, lab_img, mask=mask)

    # Calculate the average LAB values over the circular region
    avg = cv2.mean(lab_circle, mask=mask)[:3]

    # Convert the average LAB values to integers
    avg = [int(x) for x in avg]
    return avg


def get_lab_mean_from_path(path: str):
    lab_image = read_lab(path)
    avg = get_avg_lab(lab_image)

    # Convert the average LAB values to BGR color space for display
    bgr_color = cv2.cvtColor(np.array([[avg]], dtype=np.uint8), cv2.COLOR_LAB2BGR)[0][0]

    # Create an image of the color using NumPy
    color_image = np.zeros((100, 100, 3), dtype=np.uint8)
    color_image[:, :] = bgr_color

    # Display the color image using OpenCV
    cv2.imshow('LAB Color', color_image)
    cv2.waitKey(0)

    print("Average LAB values:", avg)


def get_avg_lab_from_picture(path_picture):
    img = read_img(path_picture)
    list_caps = detect_caps(img)
    for cap in list_caps:
        cap_array = cap[0]  # extract the NumPy array from the tuple
        lab_image = cv2.cvtColor(cap_array.astype(np.uint8), cv2.COLOR_BGR2LAB)
        avg = get_avg_lab(lab_image)
        print(avg)


if __name__ == '__main__':
    get_lab_mean_from_path('../database/caps-resized/5-star_200.jpg')
    get_lab_mean_from_path('../database/test-images/more-tests/9.png')

    get_lab_mean_from_path('../database/my-caps-images/9-maze.jpg')
    get_lab_mean_from_path('../database/test-images/more-tests/7.png')


