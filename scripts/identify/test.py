import torch
from dotenv import load_dotenv
from oml.models import ViTExtractor
from oml.registry import get_transforms_for_pretrained
from PIL import Image

from app.services.identify.pinecone_container import PineconeContainer

model = ViTExtractor.from_pretrained("vits16_dino").to("cpu").eval()
model.load_state_dict(torch.load('app/models/trained_model.pth'))

transform, _ = get_transforms_for_pretrained("vits16_dino")


def get_embedding(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)
    with torch.no_grad():
        embedding = model(image)

    return embedding


def test_one_vector():
    # Example usage
    image_path = 'tests/services/identify/images/1.jpg'
    embedding_vector = get_embedding(image_path)
    # print(embedding_vector)
    res = PineconeContainer().query_database(embedding_vector.tolist()[0])
    print(res)


def upload_all_vectors():


if __name__ == '__main__':
    load_dotenv()
