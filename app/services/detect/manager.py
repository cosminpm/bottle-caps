import cv2
from numpy import ndarray

from app.services.detect.blobs import get_avg_size_all_blobs
from app.services.detect.htc import hough_transform_circle

MAX_WIDTH_IMAGE = 1000
MAX_HEIGHT_IMAGE = 1000


def resize_image(src: ndarray, factor: float) -> ndarray:
    """Resize the image to a certain factor.

    Args:
    ----
        src: The source image.
        factor: The factor to resize.

    Returns:
    -------
        The resized image.

    """
    height, width = src.shape[:2]
    new_size = (int(width * factor), int(height * factor))
    return cv2.resize(src, new_size)


def crop_image_into_rectangles(photo_image: ndarray, rectangles: list) -> list[tuple]:
    """Crop the image into rectangles of the different caps.

    Args:
    ----
        photo_image: The original image.
        rectangles: The position of the rectangles.

    Returns:
    -------
        A list with the rectangles, the image and their position.

    """
    cropped_images = []
    for x, y, w, h in rectangles:
        # Sometimes we have to guarantee that rectangle size is greater than 0
        y = max(y, 0)
        x = max(x, 0)
        cropped_image = photo_image[y : y + h, x : x + w]
        if len(cropped_image) > 0:
            cropped_images.append((cropped_image, (x, y, w, h)))
    return cropped_images


def get_rectangles(circles: list) -> list:
    """Transform the circles given into rectangles.

    Args:
    ----
        circles: A list of the circles.

    Returns:
    -------
        A list with the rectangles.

    """
    rectangles = []
    for x, y, r in circles:
        x1 = x - r
        y1 = y - r
        width = r * 2
        height = r * 2
        rectangles.append((x1, y1, width, height))
    return rectangles


def preprocess_image_size(img: ndarray) -> ndarray:
    """Resize the image to a specific maximum.

    Args:
    ----
        img: The original image.

    Returns:
    -------
        The resulting resized image.

    """
    height, width = img.shape[:2]
    size = height * width
    max_size_img = MAX_WIDTH_IMAGE * MAX_HEIGHT_IMAGE
    resized = img
    while size > max_size_img:
        resized = resize_image(resized, 2 / 3)
        height, width = resized.shape[:2]
        size = height * width
    return resized


def detect_caps(img: ndarray) -> list:
    """Detect the caps in the image.

    Args:
    ----
        img: The original image.

    Returns:
    -------
        A list with the detected caps.

    """
    img = preprocess_image_size(img)

    avg_size = get_avg_size_all_blobs(img)
    cropped_images = []
    if avg_size != 0:
        circles = hough_transform_circle(img, avg_size)
        rectangles = get_rectangles(circles)
        cropped_images = crop_image_into_rectangles(img, rectangles)
    return cropped_images
