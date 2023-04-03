import json
import os

import cv2
from pathlib import Path

DEBUG_BLOB = False
MY_CAPS_IMGS_FOLDER = r"database\caps-resized"
CLUSTER_FOLDER = r"database\cluster"


def crate_db_for_cap(cap_name: str, image_folder: str, result_folder: str):
    cap_path = os.path.join(image_folder, cap_name)
    cap_img = cv2.imread(cap_path)
    cap_img = cv2.cvtColor(cap_img, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create(nfeatures=200)

    kps, dcps = sift.detectAndCompute(cap_img, None)

    keypoints_list = [[kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response, kp.octave, kp.class_id] for kp in kps]

    dcps = dcps.tolist()[:200]
    cap_name = cap_name.split(".")[0]
    cap_result = os.path.join(result_folder, cap_name) + ".json"

    entry = {
        "name": cap_name + ".jpg",
        "path": cap_path,
        "json_path": cap_result,
        "json_name": cap_name + ".json",
        "kps": keypoints_list,
        "dcps": dcps
    }

    with open(cap_result, "w") as outfile:
        # print("Writing:{}".format(cap_result))
        json.dump(entry, outfile)


def create_cap_in_database(cap_name: str, cluster: str = None):
    path = Path(os.getcwd())
    bd_folder = os.path.join(path.parent.absolute(), CLUSTER_FOLDER)
    path_caps = os.path.join(path.parent.absolute(), MY_CAPS_IMGS_FOLDER)
    if cluster is not None:
        cluster_folder = os.path.join(bd_folder, cluster)
        if not os.path.exists(cluster_folder):
            os.mkdir(cluster_folder)
    else:
        cluster_folder = bd_folder
    crate_db_for_cap(cap_name=cap_name, image_folder=path_caps, result_folder=cluster_folder)


def create_database_caps():
    """
    Create a dictionary based on the RGB values of all images

    :return: Returns the dictionary associating the path of each image and the average value of its RGB values
    """
    path = Path(os.getcwd())
    caps_folder = os.path.join(path.parent.absolute(), MY_CAPS_IMGS_FOLDER)
    entries = os.listdir(caps_folder)

    for name_img in entries:
        create_cap_in_database(cap_name=name_img)


if __name__ == '__main__':
    create_database_caps()
