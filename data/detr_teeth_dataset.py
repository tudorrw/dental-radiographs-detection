import os
import torch
from PIL import Image
import torchvision
from transformers import DetrImageProcessor, DetrFeatureExtractor
from utils.mapper import ToothLabelMapper
import albumentations as A
import numpy as np
import cv2
 
class CocoDetectionTeeth(torchvision.datasets.CocoDetection):
    """COCO-format dataset for DETR using standard COCO JSON files."""
    
    def __init__(self, json_path, image_dir, processor=None, train_mode=True):
        """
        Args:
            json_path: Path to the COCO JSON file with annotations
            image_dir: Directory with images
            processor: DetrImageProcessor instance
        """
        # Initialize the standard COCO dataset
        super().__init__(image_dir, json_path)
        self.label_mapper = ToothLabelMapper()  
        self.processor = processor
        self.train_mode = train_mode
        # self.transform = self.get_augmentations() if train_mode else None
        self.transform = None
    
    def encode_targets(self, targets):
        bboxes = []
        category_ids = []
        for target in targets:
            category_id_1 = target["category_id_1"]
            category_id_2 = target["category_id_2"]
            # Create the FDI tooth ID from the two category IDs
            tooth_id = category_id_1 * 10 + category_id_2 + 1
            # Map tooth ID to proper class index
            mapped_class_id = int(self.label_mapper.encode([tooth_id])[0])
            target["category_id"] = mapped_class_id

            bboxes.append(target["bbox"])  # Keep COCO format
            category_ids.append(mapped_class_id)
        return bboxes, category_ids


    def __getitem__(self, idx):
        """Load and process a COCO image and its annotations."""
        # Get image and annotations using the parent class method
        image, targets = super().__getitem__(idx)
        image_id = self.ids[idx]

        # Process the targets for DETR
        bboxes, category_ids = self.encode_targets(targets)
    
        if self.transform:
            image = np.array(image.convert("RGB"))
            augmented = self.transform(image=image, bboxes=bboxes, category_ids=category_ids)
            image = augmented["image"]
            bboxes = augmented["bboxes"]
            category_ids = [int(c) for c in augmented["category_ids"]] 

            # Update transformed annotations
            for i, target in enumerate(targets):
                target["bbox"] = bboxes[i]
                target["category_id"] = category_ids[i]
            image = Image.fromarray(image)
        # Format for DETR processor
        annotations = {"image_id": image_id, "annotations": targets}
        
        # Process image and annotations
        encoding = self.processor(images=image, annotations=annotations, return_tensors="pt")
        
        # Extract values needed for model training
        pixel_values = encoding["pixel_values"].squeeze()
        labels = encoding["labels"][0]

        return pixel_values, labels
    
    def get_augmentations(self): 
        return A.Compose([
            A.NoOp()
            # A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, p=0.25),
            # A.RandomGamma(gamma_limit=(80, 120), p=0.25),
            # A.CLAHE(clip_limit=2.0, tile_grid_size=(16,16), p=0.4),
        ],
        bbox_params=A.BboxParams(format='coco',label_fields=["category_ids"], clip=True)
        )

    @classmethod
    def collate_fn(cls, batch):
        """Collate function with proper padding for batching."""
        pixel_values = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        
        # Get processor for padding images properly
        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        
        # Use processor's padding capabilities to handle different image sizes
        encoding = processor.pad(pixel_values, return_tensors="pt")
        
        # Return dict format compatible with DETR model
        return {
            'pixel_values': encoding['pixel_values'],
            'pixel_mask': encoding['pixel_mask'],
            'labels': labels
        }
    


