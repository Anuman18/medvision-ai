import gradio as gr
import torch
import os
from PIL import Image
from models.feature_extractor import extract_features
from models.report_generator import ReportGenerator
from fpdf import FPDF
import uuid

model_path = "D:\\medvision-ai\\models\\cnn_model.pth"
generator = ReportGenerator()

def analyze_xray(image):
    # Save temp image
    temp_path = f"temp_{uuid.uuid4().hex}.png"
    image.save(temp_path)

    # Extract features
    features = extract_features(temp_path, model_path=model_path)

    # Generate report
    report = generator.generate(features)

    os.remove(temp_path)
    return report

def export_pdf(report_text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, report_text)
    path = f"outputs/reports/report_{uuid.uuid4().hex}.pdf"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pdf.output(path)
    return path

with gr.Blocks() as demo:
    gr.Markdown("## 🩺 MedVision AI: Chest X-ray Analyzer")

    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload Chest X-ray")
        report_output = gr.Textbox(label="Generated Medical Report", lines=10)

    generate_btn = gr.Button("Generate Report")
    pdf_btn = gr.Button("Download as PDF")
    pdf_file = gr.File(label="Download PDF")

    report_state = gr.State("")

    generate_btn.click(fn=analyze_xray, inputs=image_input, outputs=report_output)
    pdf_btn.click(fn=export_pdf, inputs=report_output, outputs=pdf_file)

demo.launch()
