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


def get_hsv(path: str):
    hsv_img = cv2.imread(path, cv2.COLOR_RGB2HSV)
    # Calculate the average values for H, S, and V
    h_avg = np.mean(hsv_img[:, :, 0])
    s_avg = np.mean(hsv_img[:, :, 1])
    v_avg = np.mean(hsv_img[:, :, 2])

    print("Hue average: ", h_avg)
    print("Saturation average: ", s_avg)
    print("Value average: ", v_avg)


def get_current_accuracy():
    test_get_dict_all_matches_i_have()


if __name__ == '__main__':
    get_hsv('../database/caps-resized/1-crown_200.jpg')
