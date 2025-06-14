import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class ChestXrayDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

        # Create a list of unique diseases
        all_labels = self.df['Finding Labels'].str.split('|').explode().unique()
        self.label_map = {label: idx for idx, label in enumerate(sorted(all_labels))}
        self.num_classes = len(self.label_map)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['Image Index'])
        image = Image.open(img_path).convert('RGB')

        labels = [0] * self.num_classes
        for disease in row['Finding Labels'].split('|'):
            if disease in self.label_map:
                labels[self.label_map[disease]] = 1

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(labels, dtype=torch.float)
