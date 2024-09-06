import cv2
import numpy as np
import torch
from torchvision import models, transforms


def initialize_resources():
    global model
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()

    global preprocess
    preprocess = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


initialize_resources()


def image_to_vector(img: np.ndarray):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = preprocess(img_rgb).unsqueeze(0)

    with torch.no_grad():
        vector = model(img_tensor)

    return vector.flatten().tolist()
