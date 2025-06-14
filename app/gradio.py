import gradio as gr
from pipeline.generate_report_from_image import generate_full_report

def analyze(image):
    findings, report = generate_full_report(image)
    return ", ".join(findings), report

gr.Interface(
    fn=analyze,
    inputs=gr.Image(type="filepath"),
    outputs=["text", "text"],
    title="🩻 MedVision AI",
    description="Upload a chest X-ray to generate a medical report."
).launch()
