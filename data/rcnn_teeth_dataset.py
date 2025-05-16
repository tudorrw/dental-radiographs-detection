import torch
from torch.utils.data import Dataset
import pandas as pd
import os
from torchvision.io import read_image
import cv2
import albumentations as A
from utils.mapper import ToothLabelMapper
class TeethDataset(Dataset):

    def __init__(self, csv_path, image_dir, dataset_type=None):
        self.data = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.dataset_type = dataset_type
        self.transform = self.get_augmentations() if dataset_type == "train" else None
        # self.transform = None
        self.label_mapper = ToothLabelMapper()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        # image_path = os.path.join(self.image_dir, sample["file_name"])
        # image = cv2.imread(image_path)
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # shape of (H, W, C)
        image = read_image(f"{self.image_dir}/{sample['file_name']}")

        # Load bounding boxes & labels
        annotations = eval(sample["annotations"])
        boxes = [ann["bbox"] for ann in annotations]
        labels = [ann["category_id_1"] * 10 + ann["category_id_2"] + 1 for ann in annotations]

        labels_encoded = self.label_mapper.encode(labels)

        # Convert to tensor
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels_encoded = torch.tensor(labels_encoded, dtype=torch.int64)

        # **Always apply resizing (even for validation)**
        if self.transform:
            image = image.permute(1, 2, 0).numpy()
            transformed = self.transform(image=image, bboxes=boxes.numpy(), category_ids=labels_encoded.numpy())
            image = torch.tensor(transformed["image"], dtype=torch.float32).permute(2, 0, 1)
            boxes = torch.tensor(transformed["bboxes"], dtype=torch.float32)
            labels_encoded = torch.tensor(transformed["category_ids"], dtype=torch.int64)

        # Convert image to tensor format (C, H, W)
        # image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        image = image[0].unsqueeze(0) / 255

        return {"image": image, "targets": dict(boxes=boxes, labels=labels_encoded), "id": sample["file_name"].split(".")[0]} 

    def get_augmentations(self):
        """Transformations for training."""
        return A.Compose([
            A.NoOp()
            # A.RandomBrightnessContrast(p=0.3),
            # A.ShiftScaleRotate(p=0.3,
            #                 shift_limit=0.1,
            #                 scale_limit=0.1,
            #                 rotate_limit=15),
            # A.CoarseDropout(num_holes_range=(5,5),
            #         hole_height_range=(70,80),
            #         hole_width_range=(70,80),
            #         fill=128,
            #         p=.25),


            
            # A.CLAHE(clip_limit=2.0, tile_grid_size=(16,16), p=0.3),
            # A.VerticalFlip(p=0.2),
        ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category_ids"]))


    @staticmethod
    def collate_fn(batch):
        """Custom collate function for DataLoader."""
        return { "image": list(map(lambda x: x["image"], batch)), 
                "targets" : list(map(lambda x: x["targets"], batch)), 
                "id": list(map(lambda x: x["id"], batch)) 
            }
