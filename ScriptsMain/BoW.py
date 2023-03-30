import json
import os

from sklearn.cluster import KMeans
import numpy as np
import matplotlib

from ScriptsMain.SIFT import get_dcp_and_kps
from ScriptsMain.utils import read_img

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def load_descs():
    script_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_folder_path = os.path.join(script_path, "database\cluster")
    descs = []
    for json_file_name in os.listdir(json_folder_path):
        json_file_path = os.path.join(json_folder_path, json_file_name)
        with open(json_file_path, 'r') as f:
            descs.append(json.load(f)['dcps'])
    return descs


def main():
    # Step 1: Extract SIFT descriptors from a sample of your changing database
    descriptors = load_descs()[:100]
    # Step 2: Cluster the descriptors into visual words using K-means clustering
    kmeans = KMeans(n_clusters=500, n_init=10)
    kmeans.fit(np.vstack(descriptors))

    # Step 3: Assign each descriptor to its nearest visual word
    bags_of_words = []

    all_dcps = load_descs()
    for dcp in all_dcps:
        labels = kmeans.predict(dcp)
        bag_of_words = np.zeros(500)
        for label in labels:
            bag_of_words[label] += 1
        bags_of_words.append(bag_of_words)

    # Step 4: Represent each image as a histogram of visual words
    histograms = []
    for bag_of_words in bags_of_words:
        histogram = bag_of_words / np.sum(bag_of_words)
        histograms.append(histogram)

    # Step 5: Compare a new SIFT descriptor to the histograms of visual words for all images in the changing database

    script_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    img = os.path.join(script_path, r"database\test-images\test-i-have\6.png")

    _, new_descriptor = get_dcp_and_kps(read_img(img))  # replace with a new SIFT descriptor
    new_descriptor = new_descriptor.astype('float')

    new_label = kmeans.predict(new_descriptor)
    new_histogram = np.zeros(500)
    new_histogram[new_label] += 1
    new_histogram /= np.sum(new_histogram)
    distances = []
    for histogram in histograms:
        distance = np.linalg.norm(new_histogram - histogram)
        distances.append(distance)


    k = 5  # Number of closest images to print
    closest_indices = np.argsort(distances)[:k]
    print(f"The {k} closest images are:")
    for i in closest_indices:
        print(i)

if __name__ == '__main__':
    main()
