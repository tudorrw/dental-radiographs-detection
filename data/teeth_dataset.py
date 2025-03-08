import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import cv2
from torchvision.transforms import ToTensor
from process.tooth_label_mapper import ToothLabelMapper

class TeethDataset(Dataset):

    def __init__(self, csv_path, image_dir, transform=None):
        self.data = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform if transform else ToTensor()
        self.label_mapper = ToothLabelMapper()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        image_path = os.path.join(self.image_dir, sample["file_name"])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load bounding boxes & labels
        annotations = eval(sample["annotations"])
        boxes = [ann["bbox"] for ann in annotations]
        labels = [ann["category_id_1"] * 10 + ann["category_id_2"] + 1 for ann in annotations]

         # Convert labels using ToothLabelMapper
        labels_encoded = self.label_mapper.encode(labels)

        # Convert to tensors
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels_encoded = torch.tensor(labels_encoded, dtype=torch.int64)

        if self.transform:
            image = self.transform(image)

        return image, dict(boxes=boxes, labels=labels_encoded)


    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function for DataLoader.
        Used to properly stack images and targets.
        """
        images, targets = zip(*batch)  # Extract images and targets
        return list(images), list(targets)
