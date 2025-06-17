# test_extract.py
from models.feature_extractor import extract_features

feature_vector = extract_features("data/chest_xrays/images/images_003/images/00003994_000.png")
print("Extracted feature shape:", feature_vector.shape)
