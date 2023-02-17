import json
import os

import cv2
import numpy as np
from pathlib import Path
from utils import set_colors

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


def crate_db_for_cap(cap_name, folder: str, cluster_folder: str):
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


def create_cap_in_database(cap_name, cluster):
    name_cluster = str(cluster)
    path = Path(os.getcwd())
    bd_folder = os.path.join(path.parent.absolute(), CLUSTER_FOLDER)
    cluster_folder = os.path.join(bd_folder, name_cluster)
    path_caps = os.path.join(path.parent.absolute(), MY_CAPS_IMGS_FOLDER)

    if not os.path.exists(cluster_folder):
        os.mkdir(cluster_folder)
    crate_db_for_cap(cap_name, path_caps, cluster_folder)


def exists_color_in_database(color):
    str_color = str(color)
    path = Path(os.getcwd())
    bd_folder = os.path.join(path.parent.absolute(), CLUSTER_FOLDER)
    color_folder = os.path.join(bd_folder, str_color)

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


def get_pixels_cap(image):
    # Obtener las dimensiones de la imagen
    height, width = image.shape[:2]

    # Calcular el radio de la circunferencia
    center = (int(width / 2), int(height / 2))
    radius = int(min(height, width) / 2)

    x_c = image.shape[1] // 2
    y_c = image.shape[0] // 2

    y, x = np.ogrid[0:image.shape[0], 0:image.shape[1]]
    mask = (x - x_c) ** 2 + (y - y_c) ** 2 <= radius ** 2
    pixels = image[mask]

    return pixels


def get_frequency_quantized_colors(image):
    # Define the set of colors to reduce to
    color_frequencies = {}
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Find the closest color in the set of colors
            color = find_closest_color(image[i, j], set_colors)
            color_tuple = tuple(color)
            if color_tuple in color_frequencies:
                color_frequencies[color_tuple] += 1
            else:
                color_frequencies[color_tuple] = 1
    return color_frequencies


def get_higher_frequency(frequencies, n=1):
    ordered_frequencies = sorted(frequencies.values(), reverse=True)
    n_most_frequent = ordered_frequencies[n - 1]  # get the n-th highest frequency
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
        create_cap_in_database(name_img, key)


def debug_one_image(path_to_image: str):
    image = read_img(path_to_image)
    # Iterate over each pixel in the image and replace its color with the closest color in the set
    color_frequencies = {}
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Find the closest color in the set of colors
            color = find_closest_color(image[i, j], set_colors)
            color_tuple = tuple(color)
            if color_tuple in color_frequencies:
                color_frequencies[color_tuple] += 1
            else:
                color_frequencies[color_tuple] = 1
            image[i, j] = color

    key = get_higher_frequency(color_frequencies)
    cv2.imshow(path_to_image, image)
    cv2.waitKey(0)


def debug_color_reduction():
    path = Path(os.getcwd())
    caps_folder = os.path.join(path.parent.absolute(), MY_CAPS_IMGS_FOLDER)
    entries = os.listdir(caps_folder)

    for name_img in entries:
        cap_str = os.path.join(caps_folder, name_img)
        debug_one_image(cap_str)


if __name__ == '__main__':
    #debug_one_image(r"C:\Users\cosmi\Desktop\BottleCaps\database\caps-s3\cap-97.jpg")
    create_database_caps()
