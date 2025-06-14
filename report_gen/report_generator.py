from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# Load pretrained T5 model (you can also try 't5-base' if RAM allows)
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Move model to GPU/CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# Example input (predicted findings from CNN or doctor)
findings = "Cardiomegaly, Effusion, Atelectasis"

# T5 input format: we give it a custom task
input_text = f"generate medical report: {findings.lower()}"

# Tokenize input
inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)

# Generate report
output = model.generate(
    inputs,
    max_length=80,
    num_beams=4,
    early_stopping=True
)

# Decode result
report = tokenizer.decode(output[0], skip_special_tokens=True)
print("\n📝 Generated Report:\n")
print(report)
