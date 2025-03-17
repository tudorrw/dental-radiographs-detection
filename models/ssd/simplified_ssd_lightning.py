import torch
import pytorch_lightning as pl
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from models.ssd.utils.simplified_ssd import SimplifiedSSD
 
 
class SimplifiedSSDLightning(pl.LightningModule):
    """PyTorch Lightning wrapper for simplified SSD model."""
    
    def __init__(self, num_classes, learning_rate, momentum):
        super().__init__()
        self.save_hyperparameters()
        
        # Model parameters
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.momentum = momentum
        
        # Initialize simplified SSD model
        self.model = SimplifiedSSD(num_classes=num_classes)
        
        # Metrics for evaluation
        self.metric = MeanAveragePrecision(
            box_format="xyxy",
            max_detection_thresholds=[50, 50, 50],  # Limit to 50 detections per image
            class_metrics=False  # Only compute overall metrics, not per-class
        )
        self.metric.warn_on_many_detections = False  # Disable the warning
        
    def forward(self, images, targets=None):
        """Forward pass through the model."""
        return self.model(images, targets)
        
    def training_step(self, batch, batch_idx):
        """Training step."""
        images, targets = batch
        batch_size = len(images)
        
        # Compute losses - torchvision SSD returns a dict of losses
        loss_dict = self.model(images, targets)
        
        # Sum all loss components
        loss = sum(loss for loss in loss_dict.values())
        
        # Log losses with explicit batch size
        self.log("train_loss", loss, prog_bar=True, batch_size=batch_size)
        for k, v in loss_dict.items():
            self.log(f"train_{k}", v, batch_size=batch_size)
            
        return loss
        
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        images, targets = batch
        batch_size = len(images)
        
        # Forward pass in eval mode
        self.model.eval()
        with torch.no_grad():
            # Get predictions - in eval mode, model returns list of dicts with 'boxes', 'labels', 'scores'
            predictions = self.model(images)
            # Calculate validation loss by temporarily setting model to train mode
            self.model.train()
            loss_dict = self.model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
            self.model.eval()
            
            # Log validation metrics
            self.log("val_loss", loss, prog_bar=True, batch_size=batch_size, sync_dist=True)
            for k, v in loss_dict.items():
                self.log(f"val_{k}", v, batch_size=batch_size, sync_dist=True)
            
            # TorchMetrics detection API wants tensors in predictions
            # The SSD model returns predictions in a different format
            formatted_predictions = []
            for pred in predictions:
                formatted_predictions.append({
                    "boxes": pred["boxes"],
                    "scores": pred["scores"],
                    "labels": pred["labels"]
                })
            
            # Update detection metrics
            self.metric.update(formatted_predictions, targets)
            
        return loss
    
    def on_validation_epoch_end(self):
        """Compute and log validation metrics at the end of each validation epoch."""
        metrics = self.metric.compute()
        
        # Log metrics
        for k, v in metrics.items():
            if not k.startswith("mar_") and k != "classes":
                self.log(f"val_{k}", v)
                
        # Reset metrics for next epoch
        self.metric.reset()
    
    def configure_optimizers(self):
        """Configure optimizers and learning rate scheduler."""
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=5e-4
        )
        
        # Use StepLR scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=5,    # Reduce LR every 5 epochs
            gamma=0.5       # Multiply by 0.5
        )
        
        return {"optimizer": optimizer, "lr_scheduler": scheduler}