import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import cv2
import albumentations as A
from utils.mapper import ToothLabelMapper
class TeethDataset(Dataset):

    def __init__(self, csv_path, image_dir, dataset_type):
        self.data = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.dataset_type = dataset_type
        self.transform = self.get_augmentations() if dataset_type == "train" else None
        self.label_mapper = ToothLabelMapper()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        image_path = os.path.join(self.image_dir, sample["file_name"])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # shape of (H, W, C)

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
            transformed = self.transform(image=image, bboxes=boxes.numpy(), category_ids=labels_encoded.numpy())
            image = transformed["image"]
            boxes = torch.tensor(transformed["bboxes"], dtype=torch.float32)
            labels_encoded = torch.tensor(transformed["category_ids"], dtype=torch.int64)

        # Convert image to tensor format (C, H, W)
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0

        return image, dict(boxes=boxes, labels=labels_encoded)

    def get_augmentations(self):
        """Transformations for training."""
        return A.Compose([
            A.HorizontalFlip(p=0.1),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category_ids"]))


    @staticmethod
    def collate_fn(batch):
        """Custom collate function for DataLoader."""
        images, targets = zip(*batch)  
        return list(images), list(targets)






# import torch
# from torch.utils.data import Dataset
# import pandas as pd
# import os
# import cv2
# import albumentations as A
# from torchvision import transforms as T
# from process.tooth_label_mapper import ToothLabelMapper

# class TeethDataset(Dataset):
#     def __init__(self, csv_path, image_dir, dataset_type):
#         self.data = pd.read_csv(csv_path)
#         self.image_dir = image_dir
#         self.dataset_type = dataset_type
#         self.label_mapper = ToothLabelMapper()

#         # Precomputed means and std (adjust if needed)
#         # self.DATA_MEANS = [0.4914, 0.4822, 0.4465]
#         # self.DATA_STD = [0.2470, 0.2435, 0.2616]

#         self.DATA_MEANS = [0.5, 0.5, 0.5]
#         self.DATA_STD = [0.5, 0.5, 0.5]

#         # Define transformations
#         self.train_transform = A.Compose([
#             A.Resize(650, 1300, p=1.0),  # Ensure uniform image size
#             A.HorizontalFlip(p=0.5),  # Random horizontal flip
#             A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
#         ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category_ids"]))

#         self.val_transform = A.Compose([
#             A.Resize(650, 1300, p=1.0),  # Ensure uniform image size
#         ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category_ids"]))

#         # Define normalization (final step applied to all images)
#         self.to_tensor = T.Compose([
#             T.ToTensor(),
#             T.Normalize(mean=self.DATA_MEANS, std=self.DATA_STD)
#         ])

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         sample = self.data.iloc[idx]
#         image_path = os.path.join(self.image_dir, sample["file_name"])
#         image = cv2.imread(image_path)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

#         # Load bounding boxes & labels
#         annotations = eval(sample["annotations"])
#         boxes = [ann["bbox"] for ann in annotations]
#         labels = [ann["category_id_1"] * 10 + ann["category_id_2"] + 1 for ann in annotations]

#         labels_encoded = self.label_mapper.encode(labels)

#         # Convert boxes & labels to tensors
#         boxes = torch.tensor(boxes, dtype=torch.float32)
#         labels_encoded = torch.tensor(labels_encoded, dtype=torch.int64)

#         # Apply augmentations (based on dataset type)
#         transform = self.train_transform if self.dataset_type == "train" else self.val_transform
#         transformed = transform(image=image, bboxes=boxes.numpy(), category_ids=labels_encoded.numpy())
#         image = transformed["image"]
#         boxes = torch.tensor(transformed["bboxes"], dtype=torch.float32)
#         labels_encoded = torch.tensor(transformed["category_ids"], dtype=torch.int64)

#         # Convert image to tensor and normalize
#         image = self.to_tensor(image)

#         return image, dict(boxes=boxes, labels=labels_encoded)

#     @staticmethod
#     def collate_fn(batch):
#         """Custom collate function for DataLoader."""
#         images, targets = zip(*batch)  # Unpack batch
        
#         # Stack images to create a proper batch
#         images = torch.stack(images, dim=0)

#         # Convert targets to a structured format (list of dictionaries -> dictionary of lists)
#         batch_targets = {"boxes": [], "labels": []}
#         for target in targets:
#             batch_targets["boxes"].append(target["boxes"])
#             batch_targets["labels"].append(target["labels"])

#         # Convert lists of tensors to batched tensors
#         batch_targets["boxes"] = torch.cat(batch_targets["boxes"], dim=0)
#         batch_targets["labels"] = torch.cat(batch_targets["labels"], dim=0)

#         return images, batch_targets


