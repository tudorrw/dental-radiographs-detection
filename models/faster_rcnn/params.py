import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pytorch_lightning as L
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.resnet import ResNet50_Weights

class TwoMLPHeadWithDropout(nn.Module):
    def __init__(self, in_channels, representation_size, dropout_prob):
        super(TwoMLPHeadWithDropout, self).__init__()
        self.dropout_prob = dropout_prob

        self.fc6 = nn.Linear(in_channels, representation_size)
        self.dropout1 = nn.Dropout(self.dropout_prob)
        self.fc7 = nn.Linear(representation_size, representation_size)
        self.dropout2 = nn.Dropout(self.dropout_prob)

    def forward(self, x):
        x = x.flatten(start_dim=1)

        x = F.relu(self.fc6(x))
        x = self.dropout1(x)
        x = F.relu(self.fc7(x))
        x = self.dropout2(x)
        return x

def add_dropout_to_backbone(model, dropout_prob=0.5):
        """
        Add dropout after each bottleneck block in the ResNet50 backbone.
        Each bottleneck has the structure:
        conv1 -> bn1 -> relu -> conv2 -> bn2 -> relu -> conv3 -> bn3 -> add identity -> relu
        
        We'll add dropout after each ReLU in the bottleneck blocks to prevent overfitting.
        """
        backbone = model.backbone.body  # This is the ResNet50 backbone
        
        # Custom bottleneck with dropout
        class BottleneckWithDropout(nn.Module):
            def __init__(self, original_bottleneck, dropout_prob):
                super(BottleneckWithDropout, self).__init__()
                
                # Copy all attributes from the original bottleneck
                self.conv1 = original_bottleneck.conv1
                self.bn1 = original_bottleneck.bn1
                self.conv2 = original_bottleneck.conv2
                self.bn2 = original_bottleneck.bn2
                self.conv3 = original_bottleneck.conv3
                self.bn3 = original_bottleneck.bn3
                self.relu = original_bottleneck.relu
                # self.downsample = original_bottleneck.downsample
                # self.stride = original_bottleneck.stride
                
                # Add dropout layer
                self.dropout = nn.Dropout(p=dropout_prob)
                
            def forward(self, x):
                identity = x
                
                # First block with dropout
                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu(out)
                out = self.dropout(out)  # Add dropout after first ReLU
                
                # Second block with dropout
                out = self.conv2(out)
                out = self.bn2(out)
                out = self.relu(out)
                out = self.dropout(out)  # Add dropout after second ReLU
                
                # Third block
                out = self.conv3(out)
                out = self.bn3(out)
                
                if self.downsample is not None:
                    identity = self.downsample(x)
                
                # Add identity connection
                out += identity
                out = self.relu(out)
                out = self.dropout(out)  # Add dropout after final ReLU
                
                return out
        
        # Replace each bottleneck with our custom version
        for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
            if hasattr(backbone, layer_name):
                layer = getattr(backbone, layer_name)
                # for i in range(len(layer)):
                    # Create a custom bottleneck with dropout
                
                original_bottleneck = layer[-1]
                layer[-1] = BottleneckWithDropout(original_bottleneck, dropout_prob)
    


model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
# add_dropout_to_backbone(model, dropout_prob=0.5)


in_features = model.roi_heads.box_predictor.cls_score.in_features

model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=33)
print(model)
# print(model.backbone.body.layer1[2])


