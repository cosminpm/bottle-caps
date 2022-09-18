import numpy as np


def window(img: np.ndarray, step_size: int, window_size: tuple):
    # slide a window across the image
    for y in range(0, img.shape[0], step_size):
        for x in range(0, img.shape[1], step_size):
            # yield the current window
            yield x, y, img[y:y + window_size[1], x:x + window_size[0]]

