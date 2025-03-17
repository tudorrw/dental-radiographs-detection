import torch
import pytorch_lightning as pl
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from models.ssd.utils.simple_ssd import SimpleSSD
 
 
class SimpleSDDLightning(pl.LightningModule):
    """
    PyTorch Lightning wrapper for SimpleSSD model.
    """
    
    def __init__(self, num_classes, learning_rate, momentum, weight_decay=0.0005):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        
        # Initialize model
        self.model = SimpleSSD(num_classes=self.num_classes)
        self._freeze_batch_form()
        # Initialize metrics
        self.metric = MeanAveragePrecision(
            box_format="xyxy",  # Format used by torchvision SSD
            class_metrics=False,  # Only compute overall metrics
            max_detection_thresholds=[50,50,50]
        )
    
    def forward(self, images, targets=None):
        """Forward pass through the model."""
        return self.model(images, targets)
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        images, targets = batch
        
        # Get losses from model
        loss_dict = self.model(images, targets)
        
        # Total loss is sum of all loss components
        loss = sum(loss for loss in loss_dict.values())
        
        # Log individual losses
        for k, v in loss_dict.items():
            self.log(f"train_{k}", v, on_step=True, on_epoch=True, prog_bar=True)
        
        # Log total loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        images, targets = batch
        
        # First get losses by running in train mode
        self.model.train()
        loss_dict = self.model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        
        # Log validation losses
        for k, v in loss_dict.items():
            self.log(f"val_{k}", v, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # Get predictions by running in eval mode
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(images)
        
        # Update metrics
        self.metric.update(predictions, targets)
        
        return loss
    
    def on_validation_epoch_end(self):
        """Compute and log metrics at the end of validation epoch."""
        metrics = self.metric.compute()
        
        # Log mean average precision metrics
        for k, v in metrics.items():
            # Skip some metrics to keep logs clean
            if not k.startswith("mar_") and k != "classes":
                self.log(f"val_{k}", v, prog_bar=True)
        
        # Reset metrics for next epoch
        self.metric.reset()
    
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        # Use SGD with momentum and weight decay - standard for SSD
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )
        
        # Learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.2,  # Reduce LR by half on plateau
            patience=5,   # Wait 5 epochs before reducing
            verbose=True
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "val_loss"  # Monitor validation loss for LR scheduling
        }
    
    def _freeze_batch_form(self):
        for module in self.model.modules():
            if isinstance(module, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
                module.eval()
                module.weight.requires_grad = False
                module.bias.requires_grad = False