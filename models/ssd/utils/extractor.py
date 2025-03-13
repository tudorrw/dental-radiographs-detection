import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights
from collections import OrderedDict


def _xavier_init(conv: nn.Module):
    """Apply Xavier initialization to Conv2D layers."""
    for layer in conv.modules():
        if isinstance(layer, nn.Conv2d):
            torch.nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, 0.0)


class SSDFeatureExtractorVGG(nn.Module):
    """VGG16 feature extractor with extra layers for SSD."""

    def __init__(self, pretrained=True):
        super().__init__()
        
        # Load VGG16 backbone (excluding fully connected layers)
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_FEATURES).features
        
        # Keep layers up to `conv5_3`
        self.features = nn.Sequential(*vgg[:30])  
        print(self.features)

        # L2 normalization on `conv4_3`
        self.scale_weight = nn.Parameter(torch.ones(512) * 20)

        # Extra feature layers for SSD (as in the SSD paper)
        self.extra_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
                nn.ReLU(inplace=True),
                nn.Conv2d(1024, 1024, kernel_size=1),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(512, 128, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3),
                nn.ReLU(inplace=True),
            )
        ])
        
        _xavier_init(self.extra_layers)

    def forward(self, x):
        """Extract feature maps for SSD head."""
        feature_maps = []
        
        # Extract feature map from VGG16
        
        x = self.features(x)
        feature_maps.append(self.scale_weight.view(1, -1, 1, 1) * F.normalize(x))

        # Extract from extra layers
        for layer in self.extra_layers:
            x = layer(x)
            feature_maps.append(x)

        return OrderedDict([(str(i), v) for i, v in enumerate(feature_maps)])
