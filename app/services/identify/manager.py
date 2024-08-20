from pathlib import Path

import keras
from keras.src.saving import load_model
from numpy import ndarray

from app.services.identify.pinecone_container import PineconeContainer, image_to_vector
from app.shared.utils import _apply_mask

PROJECT_PATH = Path.cwd()


def identify_cap(
    cap: ndarray,
    pinecone_con: PineconeContainer,
    model: keras.Sequential,
):
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


def get_model() -> keras.Sequential:
    """Get the model.

    Returns
    -------
        The keras model.

    """
    path = str(Path(PROJECT_PATH) / "app" / "models" / "model.keras")
    return load_model(path)
