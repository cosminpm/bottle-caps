import os
from pathlib import Path

from dotenv import load_dotenv

from app.services.identify.pinecone_container import PineconeContainer
from app.shared.utils import _read_img_from_path_with_mask
from scripts.identify.pretrained_model import image_to_vector

PROJECT_PATH = Path.cwd()
load_dotenv()


def generate_vector_database(
        pinecone_container: PineconeContainer
) -> None:
    """Create the vector database for pinecone connection.

    Args:
    ----
        pinecone_container: The pinecone container.
        model: The keras model

    """
    root_dir = str(Path("database") / "caps")
    folders = os.listdir(root_dir)
    for img_path in folders:
        file_path: str = str(Path(root_dir) / img_path)
        img = _read_img_from_path_with_mask(file_path)
        vector = image_to_vector(img=img)
        cap_info = {"id": img_path, "values": vector}
        pinecone_container.upsert_one_pinecone(cap_info=cap_info)


if __name__ == "__main__":
    pinecone_container = PineconeContainer()
    pinecone_container.empty_index()
    generate_vector_database(pinecone_container)
