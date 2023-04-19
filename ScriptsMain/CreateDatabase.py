import json
import os

import cv2
from pathlib import Path

from ScriptsMain.LABColor import get_avg_from_path

DEBUG_BLOB = False
MY_CAPS_IMGS_FOLDER = r"database\caps-resized"
CLUSTER_FOLDER = r"database\cluster"


PATH = Path(os.getcwd())
BD_FOLDER = os.path.join(PATH.parent.absolute(), CLUSTER_FOLDER)
CAPS_FOLDER = os.path.join(PATH.parent.absolute(), MY_CAPS_IMGS_FOLDER)


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
    avg_lab = get_avg_from_path(cap_path)
    entry = {
        "name": cap_name + ".jpg",
        "PATH": cap_path,
        "json_path": cap_result,
        "json_name": cap_name + ".json",
        "len_kps": len(kps),
        "len_dcps": len(dcps),
        "avg_lab": avg_lab,
        "kps": keypoints_list,
        "dcps": dcps,
    }

    with open(cap_result, "w") as outfile:
        print("Writing:{}".format(cap_result))
        json.dump(entry, outfile)


def create_cap_in_database(cap_name: str, cluster: str = None):
    path_caps = os.path.join(PATH.parent.absolute(), MY_CAPS_IMGS_FOLDER)
    if cluster is not None:
        cluster_folder = os.path.join(BD_FOLDER, cluster)
        if not os.path.exists(cluster_folder):
            os.mkdir(cluster_folder)
    else:
        cluster_folder = BD_FOLDER
    crate_db_for_cap(cap_name=cap_name, image_folder=path_caps, result_folder=cluster_folder)


def create_database_caps():
    """
    Create a dictionary based on the RGB values of all images

    :return: Returns the dictionary associating the PATH of each image and the average value of its RGB values
    """
    caps_folder = os.path.join(PATH.parent.absolute(), MY_CAPS_IMGS_FOLDER)
    entries = os.listdir(caps_folder)

    for name_img in entries:
        create_cap_in_database(cap_name=name_img)



def sort_database():
    entries = os.listdir(BD_FOLDER)
    for entry in entries:
        path_to_db = os.path.join(BD_FOLDER, entry)
        with open(path_to_db, "r") as file:
            json_data = json.load(file)
            json_data['avg_lab']


if __name__ == '__main__':
    sort_database()
