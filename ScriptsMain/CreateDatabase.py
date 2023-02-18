import json
import os

import cv2
import numpy as np
from pathlib import Path
from utils import colors_for_clustering

DEBUG_BLOB = False
MY_CAPS_IMGS_FOLDER = r"database\caps-s3"
DATABASE_FODLER = r"database\caps_db-s3"
CLUSTER_FOLDER = r"database\cluster"


def read_img(img_path: str) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(img_path), 1)


def rgb_to_bgr(r: int, g: int, b: int) -> tuple[int, int, int]:
    """
    Given a tuple of colors it returns the same tuple but changing the order, this is because OpenCV uses BGR instead of RGB

    :param int r: value from 0 to 255 to represent red
    :param int g: int r: value from 0 to 255 to represent green
    :param int b: int r: value from 0 to 255 to represent blu
    :return: The tuple with the three colors
    """
    return tuple((b, g, r))


def transform_bgr_image_to_rgb(img: np.ndarray) -> np.ndarray:
    """
    Transforms the image to numpy rgb from bgr

    :param np.ndarray img: The original image
    :return: The image transformed to rgb
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


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


def exists_color_in_database(color:list[tuple[int,int,int]]):
    str_color = str(color)
    path = Path(os.getcwd())
    bd_folder = os.path.join(path.parent.absolute(), CLUSTER_FOLDER)
    color_folder = os.path.join(bd_folder, str_color) + "_" + colors_for_clustering[color[0]]
    print(color_folder)
    if os.path.exists(color_folder):
        return color_folder
    return None


def find_closest_color(color, palette):
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


def get_higher_frequency(frequencies, n=1):
    ordered_frequencies = sorted(frequencies.values(), reverse=True)
    n_most_frequent = ordered_frequencies[n - 1]  # get the n-th the highest frequency
    keys = [k for k, v in frequencies.items() if v == n_most_frequent]
    return keys


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
        name_folder_cluster = str(key) + '_' + colors_for_clustering[key[0]]
        create_cap_in_database(cap_name=name_img, cluster=name_folder_cluster)


if __name__ == '__main__':
    create_database_caps()
