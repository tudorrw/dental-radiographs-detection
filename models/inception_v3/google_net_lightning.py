import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from models.inception_v3.google_net import GoogleNet

class GoogleNetPl(pl.LightningModule):
    def __init__(self, num_classes, learning_rate):
        """
        PyTorch Lightning module for GoogleNet applied to dental X-ray object detection.
        
        :param num_classes: Number of tooth classes.
        :param learning_rate: Learning rate for the optimizer.
        """
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.model = GoogleNet(num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def _compute_loss_and_acc(self, batch):
        """Helper function to compute loss and accuracy."""
        images, targets = batch
        preds = self.forward(images)

        # Ensure targets are properly structured before concatenation
        try:
            labels = [t["labels"] for t in targets if "labels" in t]
            if len(labels) == 0:  # Avoid empty tensors
                return None, None
            targets_tensor = torch.cat(labels, dim=0)  
        except Exception as e:
            raise ValueError(f"Error processing targets: {targets} -> {e}")

        loss = self.loss_fn(preds, targets_tensor)
        acc = (preds.argmax(dim=-1) == targets_tensor).float().mean()

        return loss, acc

    def training_step(self, batch, batch_idx):
        """Performs a forward pass and computes loss for training."""
        loss, acc = self._compute_loss_and_acc(batch)
        if loss is None:
            return None  # Skip empty batch issues

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Performs a forward pass and computes accuracy for validation."""
        loss, acc = self._compute_loss_and_acc(batch)
        if loss is None:
            return None  # Skip empty batch issues

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """Configures optimizer and learning rate scheduler."""
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
