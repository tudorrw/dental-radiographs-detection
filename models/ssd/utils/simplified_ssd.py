import torch
import torch.nn as nn
import torchvision.models.detection as detection
from torchvision.models.detection import ssd300_vgg16
from torchvision.models.detection.ssd import SSD300_VGG16_Weights
from torchvision.models.vgg import VGG16_Weights
 
 
class SimplifiedSSD(nn.Module):
    """SSD300 with VGG16 backbone for teeth detection in dental panoramic X-rays.
    
    This is a wrapper around torchvision's implementation, adapted for teeth detection.
    """
    
    def __init__(self, num_classes=33, pretrained=True):
        super().__init__()
        
        # Set appropriate weights for the model and backbone
        weights_backbone = VGG16_Weights.IMAGENET1K_FEATURES if pretrained else None
        
        # Create SSD model with our custom number of classes
        # Only use backbone pretrained weights (not the full SSD weights)
        # This avoids the class mismatch issue
        self.model = ssd300_vgg16(
            weights=None,  # Don't use pretrained SSD weights (avoids class count mismatch)
            weights_backbone=weights_backbone,  # Use pretrained backbone weights
            num_classes=num_classes
        )
        
        # Ensure the model handles 300x300 images consistently
        self.model.transform.min_size = [300]
        self.model.transform.max_size = 300
        
    def forward(self, images, targets=None):
        """Forward pass with both training and inference support."""
        # Convert tensor batch to list if needed
        if isinstance(images, torch.Tensor) and images.dim() == 4:
            images = list(image for image in images)
            
        # Make sure images are on the right device
        if isinstance(images, list):
            device = next(self.parameters()).device
            images = [img.to(device) for img in images]
        
        # Run the model in the correct mode
        if self.training and targets is not None:
            return self.model(images, targets)
        else:
            return self.model(images)