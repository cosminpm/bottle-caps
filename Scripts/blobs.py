import cv2
import numpy as np

DEBUG_BLOB = True


def get_avg_size_all_blobs(img: np.ndarray):
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

    return get_avg_size_blobs(keypoints)


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
