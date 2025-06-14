import torch
from torchvision import models, transforms
from PIL import Image
from transformers import T5Tokenizer, T5ForConditionalGeneration
import os
import pandas as pd

# === Load CNN ===
disease_labels = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax']

cnn_model = models.densenet121(pretrained=False)
cnn_model.classifier = torch.nn.Linear(cnn_model.classifier.in_features, len(disease_labels))
cnn_model.load_state_dict(torch.load("models/densenet_xray.pth", map_location='cpu'))
cnn_model.eval()

# === Load T5 ===
tokenizer = T5Tokenizer.from_pretrained("t5-small")
t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")
t5_model.eval()

# === Image Preprocess ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict_findings(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = cnn_model(image)
        preds = torch.sigmoid(outputs)[0]
    findings = [label for i, label in enumerate(disease_labels) if preds[i] > 0.5]
    return findings

def generate_report(findings):
    input_text = "generate medical report: " + ", ".join(findings)
    inputs = tokenizer.encode(input_text, return_tensors="pt")
    outputs = t5_model.generate(inputs, max_length=80, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# === Run full pipeline ===
def generate_full_report(image_path):
    findings = predict_findings(image_path)
    report = generate_report(findings)
    return findings, report

# === Example Usage ===
if __name__ == "__main__":
    img_path = "../data/sample_xray.jpg"  # Change as needed
    findings, report = generate_full_report(img_path)
    print("🔍 Detected Findings:", findings)
    print("📝 Report:\n", report)
