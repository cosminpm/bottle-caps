import cv2
import numpy as np
import os
from pathlib import Path


MY_CAPS_IMGS_FOLDER = r"database\caps-resized"
CLUSTER_FOLDER = r"database\cluster"

PATH = Path(os.getcwd())
CAPS_FOLDER = os.path.join(PATH.parent.absolute(), MY_CAPS_IMGS_FOLDER)
ENTRIES = os.listdir(CAPS_FOLDER)



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


def get_avg_from_path(path: str):
    lab_img = read_lab(path)
    return get_avg_lab(lab_img=lab_img)




if __name__ == '__main__':

    get_avg_lab_in_folder()