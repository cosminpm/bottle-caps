import json
import os

import cv2
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


DEBUG_BLOB = False
MY_CAPS_IMGS_FOLDER = r"database/caps-s3"
DATABASE_FODLER = r"database/caps_db-s3"


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



def plot_colors(hist, centroids):
    # initialize the bar chart representing the relative frequency
    # of each of the colors
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0
    # loop over the percentage of each cluster and the color of
    # each cluster
    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX

    # return the bar chart
    return bar


def centroid_histogram(clt):
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)
    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()
    # return the histogram
    return hist


def get_all_rgb_images():
    path = r"C:\Users\cosmi\Desktop\BottleCaps\database\caps-s3\Untitled.png"

    org_img = cv2.imread(path)
    img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)
    img = org_img.reshape((img.shape[0] * img.shape[1], 3))
    clt = KMeans(n_clusters=3)
    clt.fit(img)

    hist = centroid_histogram(clt)
    bar = plot_colors(hist, clt.cluster_centers_)
    # show our color bart
    cv2.imshow("imagen",org_img)
    plt.axis("off")
    plt.imshow(bar)
    plt.show()


def main_rgb():
    get_all_rgb_images()


if __name__ == '__main__':
    main_rgb()
    # create_json_for_all_caps()
