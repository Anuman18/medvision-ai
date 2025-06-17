# models/report_generator.py

import torch
import torch.nn as nn
from transformers import T5Tokenizer, T5ForConditionalGeneration

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReportGenerator(nn.Module):
    def __init__(self, t5_model_name='t5-small', feature_dim=1024, hidden_dim=512):
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(t5_model_name)
        self.t5 = T5ForConditionalGeneration.from_pretrained(t5_model_name)
        self.encoder_bridge = nn.Linear(feature_dim, hidden_dim)
    
    def forward(self, features, target_text=None):
        # Bridge DenseNet features to T5 encoder hidden state
        encoder_hidden_state = self.encoder_bridge(features).unsqueeze(0)  # shape: [1, hidden_dim]

        # Tokenize target
        if target_text:
            target = self.tokenizer(target_text, return_tensors="pt", padding=True, truncation=True)
            labels = target.input_ids.to(device)
        else:
            labels = None

        outputs = self.t5(
            encoder_outputs=(encoder_hidden_state.unsqueeze(1),),  # shape [batch, seq_len, hidden]
            labels=labels
        )
        return outputs

    def generate(self, image_features):
    # Convert feature vector to a sentence (you can customize this)
          prompt = "Generate a medical report for this chest X-ray image."

    # Tokenize the prompt
          inputs = self.tokenizer(prompt, return_tensors="pt")

    # Generate output from the model
          output = self.t5.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=128,
            num_beams=4,
            early_stopping=True,
    )

    # Decode generated report
          return self.tokenizer.decode(output[0], skip_special_tokens=True)