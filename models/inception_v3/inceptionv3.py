import torch
import torch.nn as nn
import torchvision.models as models
import pytorch_lightning as pl
import torch.optim as optim

class InceptionV3Lightning(pl.LightningModule):
    def __init__(self, num_classes=33, learning_rate=1e-3, pretrained=True):
        """
        PyTorch Lightning module for InceptionV3 with pretrained weights.
        
        :param num_classes: Number of tooth classes (+1 for background).
        :param learning_rate: Optimizer learning rate.
        :param pretrained: If True, loads ImageNet weights.
        """
        super(InceptionV3Lightning, self).__init__()
        self.save_hyperparameters()

        # Load pretrained InceptionV3 model
        self.model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1) if pretrained else models.inception_v3(weights=None)

        # Modify final classification layer
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

        # Optional: Modify auxiliary classifier (useful for better gradient flow)
        if self.model.aux_logits:
            aux_in_features = self.model.AuxLogits.fc.in_features
            self.model.AuxLogits.fc = nn.Linear(aux_in_features, num_classes)

        self.learning_rate = learning_rate
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        preds = self(images)

        # Handle auxiliary output (InceptionV3 returns two outputs during training)
        if isinstance(preds, tuple):
            preds, aux_preds = preds
            loss1 = self.loss_fn(preds, targets["labels"])
            loss2 = self.loss_fn(aux_preds, targets["labels"])
            loss = loss1 + 0.4 * loss2  # InceptionV3 uses weighted auxiliary loss
        else:
            loss = self.loss_fn(preds, targets["labels"])

        acc = (preds.argmax(dim=-1) == targets["labels"]).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        preds = self(images)
        loss = self.loss_fn(preds, targets["labels"])
        acc = (preds.argmax(dim=-1) == targets["labels"]).float().mean()

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
