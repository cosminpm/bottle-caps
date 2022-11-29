import cv2
import numpy as np

MATCHER = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
SIFT = cv2.SIFT_create()
MAX_MATCHES = 100


class SIFTApplied:
    def __init__(self, img):
        # Detect sift keypoints and descriptors for an img of a bottle cap
        self.kps, self.dcp = SIFT.detectAndCompute(img, None)


class SIFTMatch:
    def __init__(self, sift_cap: SIFTApplied, sift_photo: SIFTApplied):
        self.sift_cap = sift_cap
        self.sift_photo = sift_photo

        self.lst_pix = []
        self.compare_two_imgs()

    def compare_two_imgs(self) -> None:
        # Get the keypoints and descriptors
        matches = MATCHER.match(self.sift_cap.dcp, self.sift_photo.dcp)
        matches = sorted(matches, key=lambda x: x.distance)[:MAX_MATCHES]
        _, self.lst_pix = self.get_pix_kp_img(matches)

    def get_pix_kp_img(self, matches: list) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
        cap_kp_lst, photo_kp_lst = set(), set()

        for mat in matches:
            # Get the matching keypoints for each of the images
            cap_idx = mat.queryIdx
            photo_idx = mat.trainIdx

            # x - columns
            # y - rows
            # Get the coordinates
            (x1, y1) = self.sift_cap.kps[cap_idx].pt
            (x2, y2) = self.sift_photo.kps[photo_idx].pt

            cap_kp_lst.add((int(x1), int(y1)))
            photo_kp_lst.add((int(x2), int(y2)))

        return cap_kp_lst, photo_kp_lst
