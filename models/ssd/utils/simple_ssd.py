import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import ssd300_vgg16
from torchvision.models.detection.ssd import SSD300_VGG16_Weights
from torchvision.models.vgg import VGG16_Weights
 
 
class SimpleSSD(nn.Module):
    """
    A very simple SSD implementation using torchvision's built-in SSD model,
    customized for teeth detection in dental radiographs.
    """
    
    def __init__(self, num_classes, image_size=300):
        super().__init__()
        self.num_classes = num_classes
        self.image_size = image_size
        
        # Create SSD model with pretrained VGG16 backbone
        self.model = ssd300_vgg16(
            weights=None,  # No pretrained SSD weights to avoid class mismatch
            weights_backbone=VGG16_Weights.IMAGENET1K_FEATURES,  # Use pretrained backbone
            num_classes=num_classes,  # Including background class
            score_thresh=0.1,  # Lower threshold for training
            nms_thresh=0.45,
            detections_per_img=50,  # Maximum detections per image
            trainable_backbone_layers=2  # Finetune last 3 layers of backbone
        )
        
        # Don't modify the default anchor generator
        # The SSD architecture expects exactly 8732 anchor boxes with specific configurations
        # Changing this breaks the compatibility with the box coder
        
        # Instead, we can adjust detection parameters to better fit teeth detection
        self.model.box_score_thresh = 0.05  # Lower threshold to catch more candidate teeth
        self.model.box_nms_thresh = 0.35  # Adjusted for teeth separation
        
        # Ensure transform is set correctly for our image size
        self.model.transform.min_size = (image_size,)
        self.model.transform.max_size = image_size
    
    def forward(self, images, targets=None):
        """
        Forward pass handling both training and inference modes.
        
        Args:
            images: List of input images or batched tensor
            targets: Optional list of target dictionaries with 'boxes' and 'labels'
            
        Returns:
            During training: Dict with losses
            During inference: List of detection dictionaries
        """
        # Convert tensor batch to list if needed
        if isinstance(images, torch.Tensor) and images.dim() == 4:
            images = [img for img in images]
        
        # Make sure images are on the right device
        device = next(self.parameters()).device
        images = [img.to(device) for img in images]
        
        # Format targets for torchvision SSD
        if targets is not None:
            for t in targets:
                # Ensure boxes are in the right format
                if 'boxes' in t:
                    t['boxes'] = t['boxes'].to(device).float()
                if 'labels' in t:
                    t['labels'] = t['labels'].to(device).long()
        
        # Forward pass with SSD model
        if self.training:
            # During training, return the losses
            loss_dict = self.model(images, targets)
            return loss_dict
        else:
            # During inference, return the predictions
            return self.model(images)