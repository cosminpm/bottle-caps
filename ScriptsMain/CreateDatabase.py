import json
import os

import cv2
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from joblib import dump, load

DEBUG_BLOB = False
MY_CAPS_IMGS_FOLDER = r"database\caps-s3"
DATABASE_FODLER = r"database\caps_db-s3"
CLUSTER_FOLDER = r"database\cluster"


def read_img(img_path: str) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(img_path), 1)


def rgb_to_bgr(r: int, g: int, b: int) -> tuple[int, int, int]:
    return tuple((b, g, r))


def rgb_to_bgr_image(img: np.ndarray):
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


def create_circular_mask(imagen):
    high, width, _ = imagen.shape
    center = (width // 2, high // 2)
    radio = min(high, width) // 2
    mask = np.zeros((high, width), np.uint8)
    return mask, center, radio


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
        imagen_rgb = rgb_to_bgr_image(image)
        # Create a circular mask
        mask, center, radio = create_circular_mask(imagen_rgb)
        cv2.circle(mask, center, radio, (255, 255, 255), -1)
        # Apply the mask to the image
        image_mask = cv2.bitwise_and(imagen_rgb, imagen_rgb, mask=mask)
        # Format for k-means algorithm
        # b, g, r = cv2.mean(image_mask)[:3]

        red = image_mask[:, :, 0]
        green = image_mask[:, :, 1]
        blue = image_mask[:, :, 2]

        # Calculate median of RGB values
        median_r = np.median(red)
        median_g = np.median(green)
        median_b = np.median(blue)

        rgb_cap = [median_r, median_g, median_b]

        dict_rgb[cap_str] = rgb_cap

    return dict_rgb


# Create k cluster using kmeans based on the components RGB. Returns a dictionary with the clusters and their
# corresponding images
def create_clustering_rgb_kmeans():
    """
        This function performs image clustering using the k-means algorithm.

        The function takes a dictionary whose keys are the names of the images and
        whose values are the average RGB values of each image. Then, it fits the
        k-means algorithm to the RGB values and assigns each image to a cluster.

        The function returns a dictionary whose keys are the labels of the
        clusters and whose values are the images assigned to that cluster. It also
        returns the fitted kmeans object.

        :return: Returns dictionary whose keys are the labels of the clusters and whose values are the images assigned
        to that cluster and KMeans object fitted to the data.
    """

    dict_caps = get_dict_rgb_images()
    rgb_values = list(dict_caps.values())
    kmeans = KMeans(n_clusters=10, n_init=10)
    kmeans.fit(rgb_values)
    cluster_dict = {}

    for i, label in enumerate(kmeans.labels_):
        if int(label) not in cluster_dict:
            cluster_dict[int(label)] = []
        for key in dict_caps:
            if rgb_values[i] == dict_caps[key]:
                cluster_dict[int(label)].append(key)
    return cluster_dict, kmeans


def get_cluster_belong_to(kmeans, image):
    """
       This function predicts the cluster to which a given image belongs.

       The function takes a fitted k-means object and an image. It calculates
       the average RGB value of the image and uses the k-means object to predict
       the cluster to which the image belongs.

       :return: the label of the cluster to which the image belongs.
    """
    b, g, r = cv2.mean(image)[:3]
    rgb_cap = [r, g, b]
    cluster = kmeans.predict(rgb_cap)
    return cluster


def create_json_clusters_images():
    cluster_dict, model_kmeans = create_clustering_rgb_kmeans()

    path = Path(os.getcwd())
    path_folder = os.path.join(path.parent.absolute(), CLUSTER_FOLDER)
    name_model = os.path.join(path_folder, 'kmeans_model.joblib')

    dump(model_kmeans, name_model)

    name_file = os.path.join(path_folder, 'clusters.json')

    entry = cluster_dict

    with open(name_file, "w") as outfile:
        print("Writing:{}".format(path_folder))
        json.dump(entry, outfile, indent=4, separators=(',', ': '))


if __name__ == '__main__':
    # create_json_for_all_caps()
    create_json_clusters_images()
