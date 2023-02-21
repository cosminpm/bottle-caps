import json
import os

import cv2
import numpy as np
from pathlib import Path
from utils import colors_for_clustering, get_higher_frequency, read_img

DEBUG_BLOB = False
MY_CAPS_IMGS_FOLDER = r"database\caps-s3"
DATABASE_FODLER = r"database\caps_db-s3"
CLUSTER_FOLDER = r"database\cluster"


def crate_db_for_cap(cap_name: str, folder: str, cluster_folder: str):
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
    cap_result = os.path.join(cluster_folder, cap_name)

    with open(cap_result + ".json", "w") as outfile:
        print("Writing:{}".format(cap_result))
        json.dump(entry, outfile)


def create_cap_in_database(cap_name: str, cluster: str):
    path = Path(os.getcwd())
    bd_folder = os.path.join(path.parent.absolute(), CLUSTER_FOLDER)
    cluster_folder = os.path.join(bd_folder, cluster)
    path_caps = os.path.join(path.parent.absolute(), MY_CAPS_IMGS_FOLDER)

    if not os.path.exists(cluster_folder):
        os.mkdir(cluster_folder)
    crate_db_for_cap(cap_name, path_caps, cluster_folder)


def exists_color_in_database(color: list[tuple[int, int, int]]):
    str_color = str(color)
    path = Path(os.getcwd())
    bd_folder = os.path.join(path.parent.absolute(), CLUSTER_FOLDER)
    color_folder = os.path.join(bd_folder, str_color) + "_" + colors_for_clustering[color]
    print(color_folder)
    if os.path.exists(color_folder):
        return color_folder
    return None


def find_closest_color(color: np.ndarray, palette: np.ndarray):
    # Calculate the distance between the color and each color in the palette
    distances = np.sqrt(np.sum((palette - color) ** 2, axis=1))
    # Find the index of the closest color in the palette
    index = np.argmin(distances)
    # Return the closest color
    return palette[index]


def is_inside_circle(x: int, y: int, cx: int, cy: int, r: int):
    """
    Function to check if a pixel is inside a circle

    :param int x: coordinate x of the pixel
    :param int y: coordinate y of the pixel
    :param int cx: coordinate x of the center's image
    :param int cy: coordinate y of the center's image
    :param int r: radius of the circle

    :return bool: If the pixel is inside or not
    """

    distance_squared = (x - cx) ** 2 + (y - cy) ** 2
    return distance_squared <= r ** 2


def get_frequency_quantized_colors(image: np.ndarray):
    # Define the set of colors to reduce to
    color_frequencies = {}

    cx, cy = image.shape[0] // 2, image.shape[1] // 2
    r = min(cx, cy)

    list_images = np.array([k for k in colors_for_clustering.keys()])
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Find the closest color in the set of colors
            if is_inside_circle(i, j, cx, cy, r):
                color = find_closest_color(image[i, j], list_images)
                color_tuple = tuple(color)
                if color_tuple in color_frequencies:
                    color_frequencies[color_tuple] += 1
                else:
                    color_frequencies[color_tuple] = 1
    return color_frequencies


def create_database_caps():
    """
    Create a dictionary based on the RGB values of all images

    :return: Returns the dictionary associating the path of each image and the average value of its RGB values
    """
    path = Path(os.getcwd())
    caps_folder = os.path.join(path.parent.absolute(), MY_CAPS_IMGS_FOLDER)
    entries = os.listdir(caps_folder)

    for name_img in entries:
        cap_str = os.path.join(caps_folder, name_img)
        image = read_img(cap_str)

        color_frequencies = get_frequency_quantized_colors(image)

        key = get_higher_frequency(color_frequencies)
        name_folder_cluster = str(key) + '_' + colors_for_clustering[key]
        create_cap_in_database(cap_name=name_img, cluster=name_folder_cluster)


if __name__ == '__main__':
    create_database_caps()
