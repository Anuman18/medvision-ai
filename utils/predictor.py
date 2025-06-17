# utils/predictor.py
import torch
from torchvision import transforms
from PIL import Image
from transformers import T5Tokenizer, T5ForConditionalGeneration

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load CNN model
def load_cnn_model(cnn_path: str):
    from torchvision.models import densenet121
    model = densenet121(weights=None)
    model.classifier = torch.nn.Linear(model.classifier.in_features, 14)  # Update based on classes
    model.load_state_dict(torch.load(cnn_path, map_location=device))
    model.eval()
    return model.to(device)

# Load T5
def load_t5_model(model_path: str):
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    model.eval()
    return tokenizer, model.to(device)

# Predict disease from X-ray
def predict_disease(image: Image.Image, cnn_model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = cnn_model(img_tensor)
    pred = torch.sigmoid(output).cpu().numpy()[0]
    # Example classes — adjust based on your training set
    classes = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
               'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation',
               'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
    threshold = 0.5
    predictions = [classes[i] for i in range(len(classes)) if pred[i] > threshold]
    return predictions or ["No abnormality detected"]

# Generate report
def generate_report(diagnosis_list, tokenizer, t5_model):
    input_text = " ".join(diagnosis_list)
    input_ids = tokenizer.encode(f"report: {input_text}", return_tensors='pt').to(device)
    output_ids = t5_model.generate(input_ids, max_length=256, num_beams=4, early_stopping=True)
    report = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return report
