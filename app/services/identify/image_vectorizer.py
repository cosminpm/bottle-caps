import cv2
import numpy as np
import torch
from torchvision import models, transforms


class ImageVectorizer:
    def __init__(self):
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()

        self.preprocess = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def image_to_vector(self, img: np.ndarray) -> list[float]:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = self.preprocess(img_rgb).unsqueeze(0)

        with torch.no_grad():
            vector = self.model(img_tensor)
        return vector.flatten().tolist()
