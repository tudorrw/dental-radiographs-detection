import torch
import torch.nn as nn

from torchvision.ops import box_iou
from models.ssd.utils.extractor import SSDFeatureExtractorVGG, _xavier_init
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torchvision.models.detection.image_list import ImageList
class SSDHead(nn.Module):
    """SSD head for classification & regression."""

    def __init__(self, in_channels, num_anchors, num_classes):
        super().__init__()
        self.classification_head = SSDClassificationHead(in_channels, num_anchors, num_classes)
        self.regression_head = SSDRegressionHead(in_channels, num_anchors)

    def forward(self, x):
        return {
            "bbox_regression": self.regression_head(x),
            "cls_logits": self.classification_head(x),
        }


class SSDScoringHead(nn.Module):
    """Base class for classification & regression heads."""

    def __init__(self, module_list, num_columns):
        super().__init__()
        self.module_list = module_list
        self.num_columns = num_columns

    def forward(self, x):
        all_results = []
        for i, features in enumerate(x):
            results = self.module_list[i](features)
            N, _, H, W = results.shape
            results = results.view(N, -1, self.num_columns, H, W).permute(0, 3, 4, 1, 2).reshape(N, -1, self.num_columns)
            all_results.append(results)
        return torch.cat(all_results, dim=1)


class SSDClassificationHead(SSDScoringHead):
    """SSD classification head."""
    
    def __init__(self, in_channels, num_anchors, num_classes):
        cls_logits = nn.ModuleList([nn.Conv2d(c, a * num_classes, kernel_size=3, padding=1) for c, a in zip(in_channels, num_anchors)])
        _xavier_init(cls_logits)
        super().__init__(cls_logits, num_classes)


class SSDRegressionHead(SSDScoringHead):
    """SSD bounding box regression head."""
    
    def __init__(self, in_channels, num_anchors):
        bbox_reg = nn.ModuleList([nn.Conv2d(c, a * 4, kernel_size=3, padding=1) for c, a in zip(in_channels, num_anchors)])
        _xavier_init(bbox_reg)
        super().__init__(bbox_reg, 4)


class SSD(nn.Module):
    """Full SSD Model with VGG16 backbone."""

    def __init__(self, num_classes=21, image_size=(512, 512)):
        super().__init__()
        self.backbone = SSDFeatureExtractorVGG()

        self.anchor_generator = DefaultBoxGenerator(
            [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
            scales=[0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05],
            steps=[8, 16, 32, 64, 100, 300],
        )

        self.head = SSDHead([512, 1024, 512, 256, 256, 256], self.anchor_generator.num_anchors_per_location(), num_classes)

        # Add transform to correctly format images
        self.transform = GeneralizedRCNNTransform(
            min(image_size), max(image_size), image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225]
        )

    def forward(self, images):
        """Forward pass of SSD model with correct image handling."""
        
        # Convert images into ImageList format (required for anchor generator)
        image_list = ImageList(images, [(img.shape[-2], img.shape[-1]) for img in images])

        # Extract features
        features = self.backbone(image_list.tensors)

        # Generate anchors
        anchors = self.anchor_generator(image_list, list(features.values()))

        # Compute classification & regression heads
        head_outputs = self.head(list(features.values()))

        return head_outputs, anchors