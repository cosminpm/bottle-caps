from pathlib import Path

import numpy as np
from keras.src.applications.nasnet import preprocess_input
from numpy import ndarray

from app.services.identify.pinecone_container import PineconeContainer
from app.shared.utils import _apply_mask

PROJECT_PATH = Path.cwd()


def identify_cap(
    cap: ndarray,
    pinecone_con: PineconeContainer,
    model,
) -> list[dict]:
    """Identify a cap from the Pinecone database.

    Args:
    ----
        cap: The cap.
        pinecone_con: The Pinecone connection.
        model: The Keras model.

    Returns:
    -------
        The cap model with all the information.

    """
    img = _apply_mask(cap)
    vector = image_to_vector(img=img, model=model)
    result = pinecone_con.query_database(vector=vector)
    return [cap.to_dict() for cap in result]


def image_to_vector(img: ndarray, model) -> list:
    """Convert a imae into a vector.

    Args:
    ----
        img: The numpy img
        model: The keras model

    Returns:
    -------
    The vector

    """
    resized_img = np.resize(img, (224, 224, 3))
    preprocessed_img = preprocess_input(resized_img[np.newaxis, ...])
    query_feature = model.predict(preprocessed_img)
    return query_feature[0].tolist()
