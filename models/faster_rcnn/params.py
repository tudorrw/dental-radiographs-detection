import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pytorch_lightning as L
import numpy as np
import torch.nn as nn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.resnet import ResNet50_Weights

class DropoutFastRCNNPredictor(nn.Module):
    def __init__(self, in_channels, num_classes, dropout_rate=0.5):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, 1024)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(1024, 1024)
        self.cls_score = nn.Linear(1024, num_classes)
        self.bbox_pred = nn.Linear(1024, num_classes * 4)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        return scores, bbox_deltas

def create_predictor_with_dropout(in_features, num_classes):
    """
    Create a custom Fast R-CNN predictor with dropout layers.
    """
    # Create a custom box predictor with dropout
    class CustomFastRCNNPredictor(nn.Module):
        def __init__(self, in_channels, num_classes, dropout_prob):
            super(CustomFastRCNNPredictor, self).__init__()
            self.cls_score = nn.Sequential(
                nn.Dropout(p=dropout_prob),
                nn.Linear(in_channels, num_classes)
            )
            self.bbox_pred = nn.Sequential(
                nn.Dropout(p=dropout_prob),
                nn.Linear(in_channels, num_classes * 4)
            )
        
        def forward(self, x):
            if x.dim() == 4:
                # Force batch dim if x has only 3 dimensions (1, C, H, W)
                assert x.size(0) == 1
                x = x.squeeze(0)
            scores = self.cls_score(x)
            bbox_deltas = self.bbox_pred(x)
            return scores, bbox_deltas
            
    return CustomFastRCNNPredictor(in_features, num_classes, 0.5)

def add_dropout_to_mlp_head(model):
        # Get the original MLP head
        box_head = model.roi_heads.box_head
        
        # Create a custom TwoMLPHead with dropout
        class CustomTwoMLPHead(nn.Module):
            def __init__(self, original_head, dropout_prob):
                super(CustomTwoMLPHead, self).__init__()
                
                # Get the original layers
                self.original_fc6 = original_head.fc6
                self.original_fc7 = original_head.fc7
                
                # Create new sequential modules with dropout
                self.fc6 = self.original_fc6  # Keep the original linear layer
                self.dropout1 = nn.Dropout(p=dropout_prob)
                self.relu1 = nn.ReLU(inplace=True)
                
                self.fc7 = self.original_fc7  # Keep the original linear layer
                self.dropout2 = nn.Dropout(p=dropout_prob)
                self.relu2 = nn.ReLU(inplace=True)
                
                print(f"Added dropout ({dropout_prob}) before ReLU in MLP head")
            
            def forward(self, x):
                x = x.flatten(start_dim=1)
                x = self.fc6(x)
                x = self.dropout1(x)  # Add dropout before ReLU
                x = self.relu1(x)
                
                x = self.fc7(x)
                x = self.dropout2(x)  # Add dropout before ReLU
                x = self.relu2(x)
                
                return x
        
        # Replace the box_head with our custom version
        model.roi_heads.box_head = CustomTwoMLPHead(box_head, 0.25)



model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
add_dropout_to_mlp_head(model)

in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=33)
# model.roi_heads.box_predictor = create_predictor_with_dropout(in_features, num_classes=33)
print(model)

