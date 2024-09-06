import torch
import torchvision.models as models
import torchvision.transforms as transforms
import cv2


def get_pretained_model():
    resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model = torch.nn.Sequential(*list(resnet50.children())[:-1])
    model.eval()
    return model


def get_pretrained_image_vector(img_np):
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    img_tensor = preprocess(img_rgb).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        model = get_pretained_model()
        vector = model(img_tensor)
    return vector.flatten().tolist()


if __name__ == '__main__':
    image_path = 'tests/services/identify/images/1.jpg'
    img_np = cv2.imread(image_path)

    vector = get_pretrained_image_vector(img_np)
    print(len(vector))
    print(vector)
