import lightning as L
from ultralytics import YOLO
import torch
class YOLOLightning(L.LightningModule):
    def __init__(self, yaml_path, model_path):
        super().__init__()
        self.model = YOLO(yaml_path, task="detect").load(model_path)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        loss = self.model.compute_loss(images, targets)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, targets = batch
        predictions = self(images)
        val_loss = self.model.compute_loss(images, targets)
        self.log('val_loss', val_loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-4)