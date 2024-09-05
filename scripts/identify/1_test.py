import os
from pathlib import Path

import torch
from dotenv import load_dotenv
from loguru import logger
from oml.models import ViTExtractor
from oml.registry import get_transforms_for_pretrained
from PIL import Image

from app.services.identify.pinecone_container import PineconeContainer

model = ViTExtractor.from_pretrained("vits16_dino").to("cpu").eval()
model.load_state_dict(torch.load('app/models/trained_model.pth'))

transform, _ = get_transforms_for_pretrained("vits16_dino")
load_dotenv()


def get_embedding(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)
    with torch.no_grad():
        embedding = model(image)

    return embedding.tolist()[0]


def one_vector():
    image_path = 'tests/services/identify/images/1.jpg'
    vector = get_embedding(image_path)
    res = PineconeContainer().query_database(vector)
    print(res)


def upload_all_vectors():
    folder = "database/caps"
    imgs: list[str] = os.listdir(folder)
    for img in imgs:
        img_path = Path(str(folder)) / img
        cap_info = {"id": str(img_path), "values": get_embedding(img_path)}
        PineconeContainer().upsert_one_pinecone(cap_info)
        logger.info(f"Uploaded {img_path} to pinecone")


if __name__ == '__main__':
    upload_all_vectors()
