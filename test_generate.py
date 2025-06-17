# test_generate.py

from models.feature_extractor import extract_features
from models.report_generator import ReportGenerator
import torch

# Step 1: Extract features
feature_vector = extract_features("data/chest_xrays/images/images_003/images/00003994_000.png")  # path to X-ray

# Step 2: Generate report
model = ReportGenerator().to("cpu")  # change to "cuda" if GPU
model.eval()

# Step 3: Generate report
report = model.generate(feature_vector)
print("\n📝 Generated Medical Report:\n")
print(report)
