import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import cv2
import albumentations as A
from utils.mapper import ToothLabelMapper
from torchvision import transforms as T

class PanoramicDataset(Dataset):
    def __init__(self, csv_path, image_dir, dataset_type):
        self.data = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.dataset_type = dataset_type
        self.label_mapper = ToothLabelMapper()
        
        self.DATA_MEANS = [0.485, 0.456, 0.406]
        self.DATA_STD = [0.229, 0.224, 0.225]

        # **Resize all images to 512x512**
        self.train_transform = A.Compose([
            A.Resize(300, 300),  # SSD requires fixed-size input
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
        ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category_ids"]))

        self.val_transform = A.Compose([
            A.Resize(300, 300),
        ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category_ids"]))

        # Convert to tensor & normalize
        self.to_tensor = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=self.DATA_MEANS, std=self.DATA_STD)
        ])


    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        image_path = os.path.join(self.image_dir, sample["file_name"])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  

        # Load bounding boxes & labels
        annotations = eval(sample["annotations"])
        boxes = [ann["bbox"] for ann in annotations]
        labels = [ann["category_id_1"] * 10 + ann["category_id_2"] + 1 for ann in annotations]
        labels_encoded = self.label_mapper.encode(labels)

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels_encoded = torch.tensor(labels_encoded, dtype=torch.int64)

        # Apply augmentation
        transform = self.train_transform if self.dataset_type == "train" else self.val_transform
        transformed = transform(image=image, bboxes=boxes.numpy(), category_ids=labels_encoded.numpy())
        image = transformed["image"]
        boxes = torch.tensor(transformed["bboxes"], dtype=torch.float32)
        labels_encoded = torch.tensor(transformed["category_ids"], dtype=torch.int64)

        # Convert to tensor
        image = self.to_tensor(image)

        return image, {"boxes": boxes, "labels": labels_encoded}
    
    def __len__(self):
        return len(self.data)

    @staticmethod
    def collate_fn(batch):
        images, targets = zip(*batch)  # Unpack batch
        return list(images), list(targets)  # Keep as lists (SSD allows variable size)
