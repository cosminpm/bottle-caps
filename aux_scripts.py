import os

import cv2
import numpy as np

from Classes.KPsDcps import SIFTApplied


def find_dominant_color(img: np.ndarray) -> tuple[int, int, int]:
    colors = {}
    for pix in img[0]:
        pix = tuple(pix)
        pix = (pix[0] // 2, pix[1] // 2, pix[2] // 2)
        if pix not in colors:
            colors[pix] = 1
        else:
            colors[pix] += 1
    dominant = max(colors, key=colors.get)
    return tuple((int(dominant[0] * 2), int(dominant[1] * 2), int(dominant[2] * 2)))


def compare_if_same_color(c1: np.ndarray, c2: np.ndarray, ratio: float) -> bool:
    sum_numb_1 = c1[0] / (255 * 3) + c1[1] / (255 * 3) + c1[2] / (255 * 3)
    sum_numb_2 = c2[0] / (255 * 3) + c2[1] / (255 * 3) + c2[2] / (255 * 3)
    return abs(sum_numb_1 - sum_numb_2) > ratio


def distance_between_two_points(p1: tuple, p2: tuple) -> float:
    dist = ((abs(p1[0] - p2[0]) ** 2) + (abs(p1[1] - p2[1]) ** 2)) ** 0.5
    return dist


def read_img(img_path: str) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(img_path), 1)


def get_mid_point(p1: tuple[int], p2: tuple[int]) -> tuple[int, int]:
    return tuple((int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2)))


def rgb_to_bgr(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple((rgb[2], rgb[1], rgb[0]))


def get_name_from_path(path: str) -> str:
    return path.split("/")[-1]


def resize_img_pix_with_name(cap_path, path_output, pix):
    cap_name = get_name_from_path(cap_path)
    lst_name_cap = cap_name.split(".")
    cap_name = lst_name_cap[0] + "_{}".format(str(pix)) + "." + lst_name_cap[-1]
    output = resize_image(cap_path, pix, pix, path_output, cap_name)
    return output


def resize_image(path_to_image, width, height, where_save, name_output):
    src = read_img(path_to_image)
    resized = cv2.resize(src, (width, height))
    output = where_save + name_output
    cv2.imwrite(output, resized)
    return output


def resize_all_images(path, output, size):
    files = os.listdir(path)
    print(files)
    for file in files:
        resize_img_pix_with_name(path + file, output, size)


def get_kps_path(path):
    files = os.listdir(path)
    for file in files:
        img = read_img(path + file)
        print("File {} with {} kps".format(file, len(SIFTApplied(img).kps)))


def reduce_colors_images(image):
    pixels = image.reshape((-1, 3)).astype(np.float32)

    # Perform k-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness, labels, centers = cv2.kmeans(pixels, 13, None, criteria, 10, flags)

    # Convert the labels back to an image
    quantized = centers[labels]
    quantized = quantized.reshape(image.shape).astype(np.uint8)
    return quantized


def hough_transform_circle(path_to_image):
    img = cv2.imread(path_to_image, 0)
    img = cv2.GaussianBlur(img, (7, 7), 0)
    img = reduce_colors_images(img)

    # img = cv2.Canny(img, 65, 65)
    #
    # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    detector = cv2.SimpleBlobDetector_create()
    
    keypoints = detector.detect(img)
    img = cv2.drawKeypoints(img, keypoints, np.array([]), (0, 0, 255),
                            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20,
    #                            param1=50, param2=18, minRadius=20, maxRadius=90)
    #
    # circles = np.uint16(np.around(circles))

    # for i in circles[0, :]:
    #     cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
    #     cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)

    return img


if __name__ == '__main__':
    a = hough_transform_circle("./test_images/10.jpg")
    cv2.imshow("Result", a)
    cv2.waitKey(0)
    # resize_all_images("./caps_imgs/", "./resized_caps_imgs/", 200)
    # get_kps_path("./resized_caps_imgs/")
