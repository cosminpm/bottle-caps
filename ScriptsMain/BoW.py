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

file_kmeans = os.path.join(script_path, "models-bow", "model.pkl")
file_bow = os.path.join(script_path, "models-bow", "bow.json")
file_histo = os.path.join(script_path, "models-bow", "histo.json")


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
            names.append(data['json_path'])
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


def save_model_kmeans(model, filename):
    """
    Save trained model to a file.
    Args:
        model: Trained model.
        filename: Name of the file.
    """
    with open(filename, 'wb') as f:
        pickle.dump(model, f)


def load_model_kmeans(filename):
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


def train_and_save_bag_of_words(kmeans, filename_bow):
    bags_of_words = []

    all_dcps = load_descs()[1][:number_of_caps_histogram]
    for dcp in all_dcps:
        labels = kmeans.predict(dcp)
        bag_of_words = np.zeros(number_of_clusters)
        for label in labels:
            bag_of_words[label] += 1
        bags_of_words.append(bag_of_words)
    bags_of_words_list = [list(bow) for bow in bags_of_words]
    with open(filename_bow, 'w') as f:
        json.dump(bags_of_words_list, f)
    return bags_of_words


def load_bag_of_words(filename_bow):
    """
    Load bags of words from a JSON file.
    Args:
        filename_bow: Name of the file.
    Returns:
        List of bags of words.
    """
    with open(filename_bow, 'r') as f:
        bags_of_words = json.load(f)
    return bags_of_words


def train_and_save_histograms(bags_of_words, filename_histo):
    histograms = []
    for bag in bags_of_words:
        histogram = bag / np.sum(bags_of_words)
        histograms.append(histogram)

    histo_list = [list(histogram) for histogram in histograms]
    with open(filename_histo, 'w') as f:
        json.dump(histo_list, f)


def load_histogram(filename_histo):
    with open(filename_histo, 'r') as f:
        bags_of_words = json.load(f)
    return bags_of_words


def get_histogram(kmeans, descriptor):
    new_label = kmeans.predict(descriptor)
    new_histogram = np.zeros(number_of_clusters)
    new_histogram[new_label] += 1
    new_histogram /= np.sum(new_histogram)
    return new_histogram


def apply_BOW(path_to_img: str):
    kmeans = load_model_kmeans(file_kmeans)
    histograms = load_histogram(filename_histo=file_histo)

    _, new_descriptor = get_dcp_and_kps(read_img(path_to_img))  # replace with a new SIFT descriptor
    new_descriptor = new_descriptor.astype('float')

    new_histogram = get_histogram(kmeans, new_descriptor)

    distances = []
    for histogram in histograms:
        distance = np.linalg.norm(new_histogram - histogram)
        distances.append(distance)

    k = 20  # Number of closest images to print
    closest_indices = np.argsort(distances)[:k]
    print(f"The {k} closest images to {path_to_img} are:")
    names, _ = load_descs()
    names = names[:number_of_caps_histogram]
    for i in closest_indices:
        print(names[i])

    return names

def train_and_save_model():
    kmeans = KMeans(n_clusters=number_of_clusters, n_init=10)
    _, descriptors = load_descs()[:number_of_caps]
    kmeans.fit(np.vstack(descriptors))
    save_model_kmeans(kmeans, file_kmeans)
    return kmeans


def train_all():
    kmeans = train_and_save_model()
    bow = train_and_save_bag_of_words(kmeans=kmeans, filename_bow=file_bow)
    train_and_save_histograms(bags_of_words=bow, filename_histo=file_histo)


if __name__ == '__main__':
    path_to_img = os.path.join(script_path, "database", "test-images", "test-i-have", "7.png")
    apply_BOW(path_to_img=path_to_img)
