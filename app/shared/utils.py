import os
from pathlib import Path

import cv2
import numpy as np

SIFT = cv2.SIFT_create()


def _get_name_from_path(path: str) -> str:
    return path.split("\\")[-1]


def _resize_img_pix_with_name(cap_path, path_output, pix):
    cap_name = _get_name_from_path(cap_path)
    lst_name_cap = cap_name.split(".")
    cap_name = lst_name_cap[0] + f"_{pix!s}" + "." + lst_name_cap[-1]
    return _resize_image_and_save(
        path_to_image=cap_path,
        width=pix,
        height=pix,
        where_save=path_output,
        name_output=cap_name,
    )


def _read_img_from_path_with_mask(img_path: str) -> np.ndarray:
    image = cv2.imread(img_path)
    return _apply_mask(image)


def _apply_mask(image: np.ndarray) -> np.ndarray:
    height, width, _ = image.shape
    size = min(width, height)  # Take the shortest size if the image is not square
    center_x, center_y = width // 2, height // 2
    radius = size // 2

    # Create a mask with a circular region of interest (ROI) in the center
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, (center_x, center_y), radius, (255), thickness=-1)

    return cv2.bitwise_and(image, image, mask=mask)


def read_img_from_path(img_path: str) -> np.ndarray:
    """Read image from path.

    Args:
    ----
        img_path: The path of the image

    Returns:
    -------
        The numpy array of the image.

    """
    return cv2.cvtColor(cv2.imread(img_path), 1)


def img_to_numpy(img) -> np.ndarray:
    """Convert an img into a numpy array.

    Args:
    ----
        img: Image to convert

    Returns:
    -------
        The numpy img

    """
    return cv2.cvtColor(img, 1)


def _resize_image_and_save(path_to_image, width, height, where_save, name_output) -> str:
    src = read_img_from_path(path_to_image)
    resized = cv2.resize(src, (width, height))
    output = str(Path(where_save) / name_output)

    cv2.imwrite(output, resized)
    return output


def resize_all_images(path: str, output: str, size: int) -> None:
    """Resize a list of images.

    Args:
    ----
        path: Folder of the image to resize
        output: Result folder of the resized images
        size: Size of each image

    """
    files = os.listdir(path)
    for file in files:
        _resize_img_pix_with_name(str(Path(path) / file), output, size)


def rgb_to_bgr(r: int, g: int, b: int) -> tuple:
    """Reorder RGB image to BGR image.

    Args:
    ----
        r: red value
        g: green value
        b: blue value

    Returns:
    -------
    Tuple of BGR.

    """
    return b, g, r
