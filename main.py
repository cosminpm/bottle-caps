import cv2
import matplotlib.pyplot as plt
import numpy as np

# Global settings
MATCHER = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
SIFT = cv2.SIFT_create()


# Detect sift keypoints and descriptors for an img of a bottle cap
def get_kp_and_dcp(img: np.ndarray):
    kp_1, d_1 = SIFT.detectAndCompute(img, None)
    return kp_1, d_1


# Draw kp from an image path
def draw_kp(img_path: str, kp_1: tuple):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_kp = cv2.drawKeypoints(gray, kp_1, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('image', img_kp)
    cv2.waitKey(0)


def read_img(img_path):
    return cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2RGB)


def compare_two_imgs(img_cap_path, img_photo_path):
    img_cap = read_img(img_cap_path)
    img_photo = read_img(img_photo_path)

    # Get the keypoints and descriptors
    kp_cap, dcp_cap = get_kp_and_dcp(img_cap)
    kp_photo, dcp_photo = get_kp_and_dcp(img_photo)

    matches = MATCHER.match(dcp_photo, dcp_cap)
    matches = sorted(matches, key=lambda x: x.distance)

    img3 = cv2.drawMatches(img_photo, kp_photo, img_cap, kp_cap, matches[:50], img_cap, flags=2)
    plt.imshow(img3), plt.show()


if __name__ == '__main__':
    img_url = './caps_imgs/t_cap_blue.jpg'
    compare_two_imgs(img_url, './test_images/3.jpg')
    # print(type(cv2.imread(img_url)))
