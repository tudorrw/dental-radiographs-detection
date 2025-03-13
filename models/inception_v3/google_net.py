import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from types import SimpleNamespace
from models.inception_v3.inception_block import InceptionBlock

# Activation function mapping
act_fn_by_name = {
    "relu": nn.ReLU(),
}

class GoogleNet(nn.Module):
    def __init__(self, num_classes, act_fn_name="relu"):
        """
        GoogleNet-based model for dental enumeration.
        :param num_classes: Number of tooth categories (FDI system: 32 + background).
        :param learning_rate: Learning rate for optimizer.
        :param act_fn_name: Activation function (default: ReLU).
        """
        super(GoogleNet, self).__init__()
        self.num_classes = num_classes + 1
        self.act_fn = act_fn_by_name[act_fn_name]
        self.act_fn_name = act_fn_name
        # Define model parameters

        self._create_network()
        self._init_params()


    def _create_network(self):
        """Builds the GoogleNet architecture."""
        self.input_net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            self.act_fn,
        )

        # Stacking inception blocks
        self.inception_blocks = nn.Sequential(
            # Stage 1
            InceptionBlock(64, {"3x3": 32, "5x5": 16}, {"1x1": 16, "3x3": 32, "5x5": 8, "max": 8}, self.act_fn),
            InceptionBlock(64, {"3x3": 32, "5x5": 16}, {"1x1": 24, "3x3": 48, "5x5": 12, "max": 12}, self.act_fn),
            nn.MaxPool2d(3, stride=2, padding=1),  # Downsample

            # Stage 2
            InceptionBlock(96, {"3x3": 32, "5x5": 16}, {"1x1": 24, "3x3": 48, "5x5": 12, "max": 12}, self.act_fn),
            InceptionBlock(96, {"3x3": 32, "5x5": 16}, {"1x1": 16, "3x3": 48, "5x5": 16, "max": 16}, self.act_fn),
            InceptionBlock(96, {"3x3": 32, "5x5": 16}, {"1x1": 16, "3x3": 48, "5x5": 16, "max": 16}, self.act_fn),
            InceptionBlock(96, {"3x3": 32, "5x5": 16}, {"1x1": 32, "3x3": 48, "5x5": 24, "max": 24}, self.act_fn),
           
            nn.MaxPool2d(3, stride=2, padding=1),  # Downsample

            # Stage 3
            InceptionBlock(128, {"3x3": 48, "5x5": 16}, {"1x1": 32, "3x3": 64, "5x5": 16, "max": 16}, self.act_fn),
            InceptionBlock(128, {"3x3": 48, "5x5": 16}, {"1x1": 32, "3x3": 64, "5x5": 16, "max": 16}, self.act_fn),
        )

        # Fully Connected Head
        self.output_net = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Ensures consistent input size
            nn.Flatten(),
            nn.Linear(128, self.num_classes),
        )

    def _init_params(self):
        """Initializes model parameters using Kaiming He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity=self.act_fn_name)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Defines forward pass."""
        x = self.input_net(x)
        x = self.inception_blocks(x)
        x = self.output_net(x)
        return x

