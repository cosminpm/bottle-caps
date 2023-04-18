import json
import os

import numpy as np

from ScriptsMain.SIFT import get_dict_all_matches
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


def get_avg_hsv_values(path: str):
    hsv_img = read_hsv(path)
    height, width = hsv_img.shape[:2]
    center = (width // 2, height // 2)
    radius = min(center[0], center[1])

    # Create a circular mask
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    hsv_circle = cv2.bitwise_and(hsv_img, hsv_img, mask=mask)

    # Calculate the average HSV values over the circular region
    avg_hsv = cv2.mean(hsv_circle, mask=mask)[:3]

    # Convert the average HSV values to integers
    avg_hsv = [int(x) for x in avg_hsv]
    return avg_hsv


def get_hsv(path: str):
    avg_hsv = get_avg_hsv_values(path)

    # Convert the average HSV values to BGR color space for display
    bgr_color = cv2.cvtColor(np.array([[avg_hsv]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0][0]

    # Create an image of the color using NumPy
    color_image = np.zeros((100, 100, 3), dtype=np.uint8)
    color_image[:, :] = bgr_color

    # Display the color image using OpenCV
    cv2.imshow('HSV Color', color_image)
    cv2.waitKey(0)

    print("Average HSV values:", avg_hsv)


def get_current_accuracy():
    test_get_dict_all_matches_i_have()


if __name__ == '__main__':
    get_hsv('../database/caps-resized/cap-470_200.jpg')
