import os
from functools import wraps

import cv2
import numpy as np
from loguru import logger


def save_img(output_path: str, env_var_name: str = "SAVE_IMG"):
    """Save the result the image result of a function if the environment variable is on.

    Args:
    ----
        output_path (str): Where the image will be saved.
        env_var_name (str): The environment variable name.

    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            if os.getenv(env_var_name):
                if isinstance(result, np.ndarray):
                    cv2.imwrite(output_path, result)
                    logger.info(f"Array saved as image to {output_path}.")
                else:
                    logger.error("Result is not a NumPy array.")
            return result
        return wrapper
    return decorator
