import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torch
from torchvision import transforms

class ChestXrayDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        # Process the data
        self.img_paths = self.data["Image Index"].apply(lambda x: os.path.join(self.img_dir, x)).tolist()

        # Get all disease labels
        all_labels = sorted(
            list({disease for entry in self.data["Finding Labels"] for disease in entry.split('|')})
        )
        self.label_map = {label: idx for idx, label in enumerate(all_labels)}
        self.num_classes = len(self.label_map)

        self.labels = []
        for entry in self.data["Finding Labels"]:
            label_vec = [0] * self.num_classes
            for disease in entry.split('|'):
                if disease in self.label_map:
                    label_vec[self.label_map[disease]] = 1
            self.labels.append(label_vec)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            label = torch.tensor(self.labels[idx], dtype=torch.float32)

            if self.transform:
                image = self.transform(image)

            return image, label

        except Exception as e:
            print(f"[ERROR] Skipping idx {idx}, file: {img_path}, error: {e}")
            with open("bad_images.log", "a") as f:
                f.write(f"{img_path} - {str(e)}\n")

            # Return dummy data to prevent crash
            dummy_image = torch.zeros(3, 224, 224)
            dummy_label = torch.zeros(self.num_classes)
            return dummy_image, dummy_label
