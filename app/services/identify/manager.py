from pathlib import Path

import torch
from numpy import ndarray
from torch import nn

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


def get_oml_model() -> nn.Module:
    """Load the OML model from a .pth file."""
    path = str(Path(PROJECT_PATH) / "app" / "models" / "trained_model.pth")
    state_dict = torch.load(path)

    # Initialize your OML model (replace CustomOMLModel with your actual OML model class)
    model = CustomOMLModel()
    model.load_state_dict(state_dict)

    # Set to evaluation mode (important to disable dropout/batchnorm during inference)
    model.eval()

    return model
