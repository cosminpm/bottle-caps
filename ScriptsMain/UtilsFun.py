import os
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

SIFT = cv2.SIFT_create()


def get_name_from_path(path: str) -> str:
    return path.split("\\")[-1]


def resize_img_pix_with_name(cap_path, path_output, pix):
    cap_name = get_name_from_path(cap_path)
    lst_name_cap = cap_name.split(".")
    cap_name = lst_name_cap[0] + "_{}".format(str(pix)) + "." + lst_name_cap[-1]
    output = resize_image_and_save(path_to_image=cap_path, width=pix, height=pix, where_save=path_output,
                                   name_output=cap_name)
    return output


def read_img_from_path_with_mask(img_path: str) -> np.ndarray:
    image = cv2.imread(img_path)
    return _apply_mask(image)


def read_img_with_mask(image: np.ndarray) -> np.ndarray:
    return _apply_mask(image)


def _apply_mask(image: np.ndarray) -> np.ndarray:
    height, width, _ = image.shape
    size = min(width, height)  # Take the shortest size if the image is not square
    center_x, center_y = width // 2, height // 2
    radius = size // 2

    # Create a mask with a circular region of interest (ROI) in the center
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, (center_x, center_y), radius, (255), thickness=-1)

    # Apply the mask to the image
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    return masked_image


def read_img_from_path(img_path: str) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(img_path), 1)


def img_to_numpy(img_np) -> np.ndarray:
    return cv2.cvtColor(img_np, 1)


def get_dcp_and_kps(img: np.ndarray) -> Tuple:
    """
    Detect and compute the descriptors and keypoints of the image

    ":param np.ndarray img: The image to get descriptors and keypoints
    :return: Returns a tuple with descriptors and keypoints
    """
    return SIFT.detectAndCompute(img, None)


def resize_image_and_save(path_to_image, width, height, where_save, name_output):
    src = read_img_from_path(path_to_image)
    resized = cv2.resize(src, (width, height))
    output = os.path.join(where_save, name_output)

    cv2.imwrite(output, resized)
    return output


def resize_all_images(path, output, size):
    files = os.listdir(path)
    for file in files:
        resize_img_pix_with_name(os.path.join(path, file), output, size)


# -- Finish resize


def rgb_to_bgr(r: int, g: int, b: int) -> Tuple:
    """
    Given a tuple of colors it returns the same tuple but changing the order, this is because OpenCV uses BGR instead of RGB

    :param int r: value from 0 to 255 to represent red
    :param int g: int r: value from 0 to 255 to represent green
    :param int b: int r: value from 0 to 255 to represent blu
    :return: The tuple with the three colors
    """
    return tuple((b, g, r))


def main():
    cwd = Path(os.getcwd())
    path_1 = os.path.join(cwd.parent.absolute(), r'database\1000-caps-s3-images')
    path_2 = os.path.join(cwd.parent.absolute(), r'database\my-caps-images')
    output = os.path.join(cwd.parent.absolute(), r'database\caps-resized')

    resize_all_images(path=path_1, output=output, size=150)
    resize_all_images(path=path_2, output=output, size=150)


if __name__ == '__main__':
    main()
