import cv2
import numpy as np

DEBUG_BLOB = False
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


def get_avg_size_all_blobs(img: np.ndarray):
    img = preprocess_image_blobs(img)
    params = cv2.SimpleBlobDetector_Params()

    params.filterByArea = True
    params.minArea = 100
    params.maxArea = img.shape[0] * img.shape[1] * (9 / 10)

    detector = cv2.SimpleBlobDetector_create(params)

    keypoints = detector.detect(img)
    keypoints = remove_overlapping_blobs(kps=keypoints)

    if DEBUG_BLOB:
        img = cv2.drawKeypoints(img, keypoints, np.array([]), (0, 0, 255),
                                cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow("Result", img)
        cv2.waitKey(0)

    return int(get_avg_size_blobs(keypoints)/2)


def get_avg_size_blobs(kps: list[cv2.KeyPoint]):
    total_radius = 0
    for kp in kps:
        total_radius += kp.size

    avg_radius = total_radius / len(kps)
    return avg_radius


def remove_overlapping_blobs(kps: list[cv2.KeyPoint]):
    boxes = []
    for kp in kps:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        r = int(kp.size / 2)
        box = (x - r, y - r, x + r, y + r)
        boxes.append(box)

    # Iterate over the blobs and mark overlapping ones
    overlapping = [False] * len(kps)
    for i, box in enumerate(boxes):
        if overlapping[i]:
            continue
        for j, other_box in enumerate(boxes[i + 1:]):
            if box[0] < other_box[2] and box[2] > other_box[0] and box[1] < other_box[3] and box[3] > other_box[1]:
                overlapping[i + j + 1] = True

    # Keep only the non-overlapping blobs
    filtered_keypoints = []
    for i, kp in enumerate(kps):
        if not overlapping[i]:
            filtered_keypoints.append(kp)
    return filtered_keypoints
