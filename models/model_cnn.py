# models/model_cnn.py
import torch
from torchvision.models import densenet121

def get_model(num_classes=15):
    model = densenet121(weights=None)
    model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
    return model
