import json
import os

import cv2
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from joblib import dump
from blobs import reduce_colors_images

DEBUG_BLOB = False
MY_CAPS_IMGS_FOLDER = r"database\caps-s3"
DATABASE_FODLER = r"database\caps_db-s3"
CLUSTER_FOLDER = r"database\cluster"


def read_img(img_path: str) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(img_path), 1)


def rgb_to_bgr(r: int, g: int, b: int) -> tuple[int, int, int]:
    return tuple((b, g, r))


def bgr_to_rgb_image(img: np.ndarray):
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

    with open('../' + cap_result + ".json", "w") as outfile:
        print("Writing:{}".format(cap_result))
        json.dump(entry, outfile)


def create_json_for_all_caps():
    path = Path(os.getcwd())
    path_caps = os.path.join(path.parent.absolute(), MY_CAPS_IMGS_FOLDER)

    entries = os.listdir(path_caps)
    for name_img in entries:
        crate_db_for_cap(name_img, path_caps)

def find_closest_color(color, palette):
    # Calculate the distance between the color and each color in the palette
    distances = np.sqrt(np.sum((palette - color) ** 2, axis=1))
    # Find the index of the closest color in the palette
    index = np.argmin(distances)
    # Return the closest color
    return palette[index]


def get_dict_rgb_images():
    """
    Create a dictionary based on the RGB values of all images

    :return: Returns the dictionary associating the path of each image and the average value of its RGB values
    """
    path = Path(os.getcwd())
    caps_folder = os.path.join(path.parent.absolute(), MY_CAPS_IMGS_FOLDER)
    entries = os.listdir(caps_folder)
    dict_rgb = {}

    for name_img in entries:
        cap_str = os.path.join(caps_folder, name_img)
        image = read_img(cap_str)

        # Define the set of colors to reduce to

        colors  =  np.array([
            [255, 255, 255],  # white
            [0, 0, 0],  # black
            [128, 128, 128],  # gray
            [255, 0, 0],  # red
            [255, 128, 0],  # orange
            [255, 255, 0],  # yellow
            [128, 255, 0],  # lime green
            [0, 255, 0],  # green
            [0, 255, 128],  # spring green
            [0, 255, 255],  # cyan
            [0, 128, 255],  # azure
            [0, 0, 255],  # blue
            [127, 0, 255],  # purple
            [255, 0, 255],  # magenta
            [255, 51, 153],  # rose
            [204, 153, 255],  # lavender
            [102, 0, 51],  # maroon
            [153, 51, 0],  # brown
            [255, 153, 51],  # coral
            [255, 204, 153],  # peach
            [255, 255, 153],  # pale yellow
            [204, 255, 153],  # pale green
            [153, 255, 153],  # pale lime green
            [153, 255, 204],  # pale spring green
            [153, 255, 255],  # pale cyan
            [153, 204, 255],  # pale azure
            [153, 153, 255],  # pale blue
            [204, 153, 255],  # pale purple
            [255, 153, 255],  # pale magenta
            [255, 204, 229],  # pale rose
            [229, 204, 255],  # pale lavender
            [127, 51, 0],  # dark brown
            [255, 0, 102],  # dark pink
            [51, 51, 0],  # olive
            [0, 51, 51],  # teal
            [0, 0, 51],  # navy blue
            [102, 0, 102],  # dark purple
            [102, 0, 51],  # dark red
            [153, 102, 51],  # dark tan
            [102, 102, 153],  # slate blue
            [51, 102, 153],  # steel blue
            [153, 51, 102],  # dark magenta
            [102, 51, 102],  # dark violet
            [102, 153, 51],  # dark lime green
            [51, 153, 102],  # dark sea green
            [153, 51, 51],  # dark salmon
            [51, 153, 153]  # dark turquoise
        ])
        # Iterate over each pixel in the image and replace its color with the closest color in the set
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                # Find the closest color in the set of colors
                color = find_closest_color(image[i, j], colors)
                # Replace the pixel's color with the closest color
                image[i, j] = color

        cv2.imshow(name_img, image)
        cv2.waitKey(0)

        print(image)
        # Calculate median of RGB values
        median = np.median(image)
        print(median)

    return dict_rgb

if __name__ == '__main__':
    # create_json_for_all_caps()
    # create_json_clusters_images()
    get_dict_rgb_images()
