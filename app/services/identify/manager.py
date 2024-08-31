from pathlib import Path
from typing import Any

import torch
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

def image_to_vector(img, model):
    model.eval()
    return torch.from_numpy(img).float()


def get_model() -> Any:
    """Get the model.

    Returns
    -------
        The keras model.

    """
    path = str(Path(PROJECT_PATH) / "app" / "models" / "trained_model.pth")
    state_dict = torch.load(path)
    model.load_state_dict(state_dict)
    return model
