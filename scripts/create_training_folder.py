import os
from pathlib import Path

import cv2
import numpy as np

from app.shared.utils import _read_img_from_path_with_mask


def apply_modifications(img):
    """Apply modifications such as rotation and color adjustments to the image."""

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue_shift = np.random.randint(-10, 10)
    hsv[..., 0] = (hsv[..., 0] + hue_shift) % 180
    img_color_shifted = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    rows, cols, _ = img.shape
    angle = np.random.uniform(0, 360)
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    img_rotated = cv2.warpAffine(img_color_shifted, M, (cols, rows))

    return img_rotated


def create_img_training(name: str, folder_create: str, path_all_images: str) -> None:
    """Create the training image for the model."""
    folder_name = Path(name).stem
    folder_result = Path(folder_create) / folder_name

    if not folder_result.exists():
        folder_result.mkdir(parents=True)

    path_img = Path(path_all_images) / name
    img = _read_img_from_path_with_mask(str(path_img))

    cv2.imwrite(str(folder_result / name), img)

    for i in range(3):  # Create 3 variations
        modified_img = apply_modifications(img)
        modified_img_name = f"{folder_name}_mod_{i}.jpg"
        cv2.imwrite(str(folder_result / modified_img_name), modified_img)


def create_training_folder() -> None:
    """Create the training folder that's going to be used to train the model."""
    path_all_images = str(Path("database") / "caps")
    folder_create = str(Path("database") / "training")

    names_images = os.listdir(path=path_all_images)
    for name in names_images:
        create_img_training(name=name, folder_create=folder_create, path_all_images=path_all_images)


if __name__ == '__main__':
    create_training_folder()
