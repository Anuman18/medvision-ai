import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
from utils.dataloader import ChestXrayDataset

from tqdm import tqdm
import os

# Paths
CSV_PATH = "D:\medvision-ai\data\chest_xrays\Data_Entry_2017.csv"
IMG_DIR = "D:\medvision-ai\data\chest_xrays\images"
SAVE_PATH = "models/densenet_xray.pth"

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Dataset and Dataloader
dataset = ChestXrayDataset(CSV_PATH, IMG_DIR, transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Model: DenseNet121
model = models.densenet121(pretrained=True)
model.classifier = nn.Linear(model.classifier.in_features, dataset.num_classes)
model = model.to(device)

# Loss and Optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop (5 epochs)
for epoch in range(5):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(dataloader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/5 - Loss: {running_loss:.4f}")
    

# Save the model
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), SAVE_PATH)
print("✅ Model saved at:", SAVE_PATH)
