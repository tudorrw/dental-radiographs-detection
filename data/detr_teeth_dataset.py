import os
import torch
from PIL import Image
import torchvision
from transformers import DetrImageProcessor
from utils.mapper import ToothLabelMapper
 
class CocoDetectionTeeth(torchvision.datasets.CocoDetection):
    """COCO-format dataset for DETR using standard COCO JSON files."""
    
    def __init__(self, json_path, image_dir, processor=None):
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
    
    def __getitem__(self, idx):
        """Load and process a COCO image and its annotations."""
        # Get image and annotations using the parent class method
        image, targets = super().__getitem__(idx)
        image_id = self.ids[idx]

        # Process the targets for DETR
        for target in targets:
            category_id_1 = target.get("category_id_1", 0)
            category_id_2 = target.get("category_id_2", 0)
            # Create the FDI tooth ID from the two category IDs
            tooth_id = category_id_1 * 10 + category_id_2 + 1
            # Map tooth ID to proper class index
            mapped_class_id = int(self.label_mapper.encode([tooth_id])[0])
            # target["class_labels"] = torch.tensor(mapped_class_id, dtype=torch.long)
            target["category_id"] = mapped_class_id

            # Ensure bounding boxes are tensors
            # target["boxes"] = torch.tensor(target["bbox"], dtype=torch.float32)
            # print("image id: ", image_id, "target: ", category_id_1 * 10 + category_id_2 + 1, "mapped_class_id: ", mapped_class_id)  
        # Format for DETR processor
        annotations = {"image_id": image_id, "annotations": targets}
        
        # Process image and annotations
        encoding = self.processor(images=image, annotations=annotations, return_tensors="pt")
        
        # Extract values needed for model training
        pixel_values = encoding["pixel_values"].squeeze()
        labels = encoding["labels"][0]
        return pixel_values, labels
    
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