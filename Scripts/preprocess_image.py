import cv2
import numpy as np

DEBUG_PREPROCESS_BLOBS = False


def reduce_colors_images(image, number_of_levels):
    pixels = image.reshape((-1, 3)).astype(np.float32)

    # Perform k-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness, labels, centers = cv2.kmeans(pixels, number_of_levels, None, criteria, 10, flags)

    # Convert the labels back to an image
    quantized = centers[labels]
    quantized = quantized.reshape(image.shape).astype(np.uint8)
    return quantized


def preprocess_image_blobs(img):
    img = cv2.GaussianBlur(img, (15, 15), 0)
    img = reduce_colors_images(img, 3)
    if DEBUG_PREPROCESS_BLOBS:
        cv2.imshow("Preprocess img", img)
        cv2.waitKey(0)
    return img
