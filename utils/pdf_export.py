from fpdf import FPDF

def save_report_pdf(image_path, findings, report, output_path="output_report.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="MedVision AI Report", ln=1, align="C")
    pdf.image(image_path, x=30, y=30, w=150)

    pdf.set_xy(10, 120)
    pdf.multi_cell(0, 10, f"Detected Findings: {', '.join(findings)}\n\nGenerated Report:\n{report}")

    pdf.output(output_path)
    print(f"✅ PDF saved to {output_path}")
