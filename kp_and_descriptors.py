import cv2
import numpy as np

MATCHER = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
SIFT = cv2.SIFT_create()
MAX_MATCHES = 35


# Detect sift keypoints and descriptors for an img of a bottle cap
def get_kp_and_dcp(img: np.ndarray):
    kp_1, d_1 = SIFT.detectAndCompute(img, None)
    return kp_1, d_1


def compare_two_imgs(img_cap: np.ndarray, img_photo: np.ndarray):
    # Get the keypoints and descriptors
    kp_cap, dcp_cap = get_kp_and_dcp(img_cap)
    kp_photo, dcp_photo = get_kp_and_dcp(img_photo)

    matches = MATCHER.match(dcp_cap, dcp_photo)
    matches = sorted(matches, key=lambda x: x.distance)[:MAX_MATCHES]
    _, lst_pix = get_pix_kp_img(matches, kp_cap, kp_photo)
    return lst_pix


def get_pix_kp_img(matches: list, cap_kp: tuple, photo_kp: tuple):
    cap_kp_lst, photo_kp_lst = set(), set()

    for mat in matches:
        # Get the matching keypoints for each of the images
        cap_idx = mat.queryIdx
        photo_idx = mat.trainIdx

        # x - columns
        # y - rows
        # Get the coordinates
        (x1, y1) = cap_kp[cap_idx].pt
        (x2, y2) = photo_kp[photo_idx].pt

        cap_kp_lst.add((int(x1), int(y1)))
        photo_kp_lst.add((int(x2), int(y2)))

    return cap_kp_lst, photo_kp_lst
