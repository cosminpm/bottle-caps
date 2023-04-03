import json
import os
import pickle

import numpy as np
from sklearn.cluster import KMeans

from ScriptsMain.SIFT import get_dcp_and_kps
from ScriptsMain.utils import read_img

script_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
number_of_clusters = 150
number_of_caps = 300
number_of_caps_histogram = 500

def load_descs():
    """
    Load the descriptors from JSON files.
    Returns:
        names: List of image names.
        descs: List of descriptors.
    """
    json_folder_path = os.path.join(script_path, "database\cluster")
    descs = []
    names = []
    for json_file_name in os.listdir(json_folder_path):
        json_file_path = os.path.join(json_folder_path, json_file_name)
        with open(json_file_path, 'r') as f:
            data = json.load(f)
            names.append(data['name'])
            descs.append(data['dcps'])
    return names, descs


def train_kmeans(descriptors, n_clusters=number_of_clusters, n_init=10):
    """
    Train K-means clustering model using descriptors.
    Args:
        descriptors: List of descriptors.
        n_clusters: Number of clusters.
        n_init: Number of times the k-means algorithm will be run with different centroid seeds.
    Returns:
        trained KMeans model.
    """
    kmeans = KMeans(n_clusters=n_clusters, n_init=n_init)
    kmeans.fit(np.vstack(descriptors))
    return kmeans


def save_model(model, filename):
    """
    Save trained model to a file.
    Args:
        model: Trained model.
        filename: Name of the file.
    """
    with open(filename, 'wb') as f:
        pickle.dump(model, f)


def load_model(filename):
    """
    Load trained model from a file.
    Args:
        filename: Name of the file.
    Returns:
        Trained model.
    """
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    return model


def main():

    # Step 2: Cluster the descriptors into visual words using K-means clustering
    file_kmeans = os.path.join(script_path, "models-bow\model.pkl")
    kmeans = load_model(file_kmeans)

    # Step 3: Assign each descriptor to its nearest visual word
    bags_of_words = []

    all_dcps = load_descs()[1][:number_of_caps_histogram]
    for dcp in all_dcps:
        labels = kmeans.predict(dcp)
        bag_of_words = np.zeros(number_of_clusters)
        for label in labels:
            bag_of_words[label] += 1
        bags_of_words.append(bag_of_words)

    # Step 4: Represent each image as a histogram of visual words
    histograms = []
    for bag_of_words in bags_of_words:
        histogram = bag_of_words / np.sum(bag_of_words)
        histograms.append(histogram)

    # Step 5: Compare a new SIFT descriptor to the histograms of visual words for all images in the changing database

    img = os.path.join(script_path, r"database\test-images\test-i-have\6.png")

    _, new_descriptor = get_dcp_and_kps(read_img(img))  # replace with a new SIFT descriptor
    new_descriptor = new_descriptor.astype('float')

    new_label = kmeans.predict(new_descriptor)
    new_histogram = np.zeros(number_of_clusters)
    new_histogram[new_label] += 1
    new_histogram /= np.sum(new_histogram)
    distances = []
    for histogram in histograms:
        distance = np.linalg.norm(new_histogram - histogram)
        distances.append(distance)

    k = 50  # Number of closest images to print
    closest_indices = np.argsort(distances)[:k]
    print(f"The {k} closest images are:")
    names, _ = load_descs()
    names = names[:number_of_caps_histogram]
    for i in closest_indices:
        print(names[i])
    print("LEN", len(distances))


def train_and_save_model():
    kmeans = KMeans(n_clusters=number_of_clusters, n_init=10)
    _, descriptors = load_descs()[:number_of_caps]
    kmeans.fit(np.vstack(descriptors))
    file_kmeans = os.path.join(script_path, "models-bow", "model.pkl")
    save_model(kmeans, file_kmeans)



if __name__ == '__main__':
    train_and_save_model()
    main()
