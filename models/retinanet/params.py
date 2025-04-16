import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import pytorch_lightning as L
import numpy as np
from functools import partial
import torchvision
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from torchvision.models.detection import RetinaNet_ResNet50_FPN_Weights, RetinaNet_ResNet50_FPN_V2_Weights


model = torchvision.models.detection.retinanet_resnet50_fpn(
            weights=RetinaNet_ResNet50_FPN_Weights.COCO_V1
        )
num_anchors = model.head.classification_head.num_anchors

model.head.classification_head = RetinaNetClassificationHead(
    in_channels=256,
    num_anchors=num_anchors,
    num_classes=2,
    norm_layer=partial(torch.nn.GroupNorm, 32)
)
print(model)