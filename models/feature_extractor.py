# models/feature_extractor.py
import torch
import torchvision.transforms as transforms
from PIL import Image
from .model_cnn import get_model
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_features(image_path, model_path="models/cnn_model.pth"):
    # Load model
    model = get_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval().to(device)

    # Remove the final classification layer to get features
    model.classifier = torch.nn.Identity()

    # Load image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])

    image = Image.open(image_path).convert("RGB")  # ✅ Convert to 3-channel image
    image = transform(image)
    image = image.unsqueeze(0).to(device)

    # Extract features
    with torch.no_grad():
        features = model(image)  # Shape: [1, 1024]

    return features.squeeze(0)  # Shape: [1024]
