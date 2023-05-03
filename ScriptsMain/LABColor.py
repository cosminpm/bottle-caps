import heapq
import json
from math import sqrt

import cv2
import numpy as np

MAX_DISTANCE = 150
FULL_PATH_SORTED_CLUSTER_FILE = r'C:\Users\cosmi\Desktop\BottleCaps\database\sorted_cluster.json'

def read_lab(path: str):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2LAB)


def read_lab_from_np(image_np: np.ndarray):
    return cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)


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


def get_avg_lab_from_np(img_np: np.ndarray):
    lab_img = read_lab_from_np(img_np)
    return get_avg_lab(lab_img=lab_img)


def get_avg_lab_from_path(path: str):
    lab_img = read_lab(path)
    return get_avg_lab(lab_img=lab_img)


def delta_e(color1, color2):
    l = (color1[0] - color2[0]) ** 2 * 0.00
    a = (color1[1] - color2[1]) ** 2 * 0.95
    b = (color1[2] - color2[2]) ** 2 * 0.05
    result = sqrt(l + a + b)
    return result


def find_closest_matches(lst, new_tuple):
    """
    Finds the indices of the 10 closest matches to a new tuple in a sorted list of tuples.
    """
    max_distance = MAX_DISTANCE
    heap = []
    for index, color_lst in enumerate(lst):
        diff = delta_e(color_lst, new_tuple)
        if len(heap) < max_distance:
            heapq.heappush(heap, (-diff, index))
        elif -diff > heap[0][0]:
            heapq.heappushpop(heap, (-diff, index))

    closest_tuples = sorted(heap, key=lambda x: -x[0])
    print(closest_tuples)
    closest_indices = [index for _, index in closest_tuples]
    return closest_indices


def find_closest_match_in_cluster_json(path: str, color_to_be_found: tuple) -> list[int]:
    with open(path, "r") as file:
        json_data = json.load(file)
        lst = [(i['avg_lab'][0], i['avg_lab'][1], i['avg_lab'][2]) for i in json_data]
        best_matches = find_closest_matches(lst, color_to_be_found)
        return [json_data[index] for index in best_matches]


def display_lab_color(lab_color: tuple[int, int, int]):
    # Create a single pixel image with the LAB color
    lab_image = np.zeros((100, 100, 3), dtype=np.uint8)
    lab_image[..., 0] = lab_color[0]  # L channel
    lab_image[..., 1] = lab_color[1]  # a channel
    lab_image[..., 2] = lab_color[2]  # b channel

    # Convert the LAB image to RGB for display
    rgb_image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)

    # Show the RGB image using OpenCV
    cv2.imshow("LAB Color", rgb_image)
    cv2.waitKey(0)


if __name__ == '__main__':

    color = (190, 131, 157)
    #a = get_avg_lab_from_path(r'C:\Users\cosmi\Desktop\BottleCaps\database\test-images\one-image\img.png')
    best_match = find_closest_match_in_cluster_json(FULL_PATH_SORTED_CLUSTER_FILE, color)
    print(best_match)
