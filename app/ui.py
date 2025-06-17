# app/ui.py
import gradio as gr
from utils.predictor import load_cnn_model, load_t5_model, predict_disease, generate_report
from PIL import Image

cnn_model = load_cnn_model("models/cnn_model.pth")
tokenizer, t5_model = load_t5_model("models/report_t5_model")

def process_image(image):
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    diagnoses = predict_disease(image, cnn_model)
    report = generate_report(diagnoses, tokenizer, t5_model)
    return ", ".join(diagnoses), report

iface = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="pil", label="Upload Chest X-ray"),
    outputs=[
        gr.Textbox(label="Predicted Diagnoses"),
        gr.Textbox(label="Generated Medical Report")
    ],
    title="MedVision AI",
    description="Upload a chest X-ray to get a diagnosis and medical report."
)

if __name__ == "__main__":
    iface.launch()
