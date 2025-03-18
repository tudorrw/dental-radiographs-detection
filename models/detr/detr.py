 
import torch
import pytorch_lightning as pl
from transformers import DetrForObjectDetection, DetrImageProcessor
from torchmetrics.detection.mean_ap import MeanAveragePrecision
 
class DETR(pl.LightningModule):
    def __init__(self, num_classes, learning_rate=1e-5, weight_decay=1e-4):
        super().__init__()
        
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Load pre-trained DETR model
        self.model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50",
            num_labels=self.num_classes,
            ignore_mismatched_sizes=True
        )
        
        # Metrics
        self.metric = MeanAveragePrecision(box_format="xywh")
        
    def forward(self, pixel_values, pixel_mask=None, labels=None):
        return self.model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            labels=labels
        )
    
    def common_step(self, batch, batch_idx):
        pixel_values = batch['pixel_values']
        pixel_mask = batch['pixel_mask']
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch['labels']]
        
        # Forward pass
        outputs = self.model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            labels=labels
        )
        
        loss = outputs.loss
        loss_dict = outputs.loss_dict
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        # Extract from dict format
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("training_loss", loss)
        # for k,v in loss_dict.items():
        #     self.log(f"train_{k}", v)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # Extract from dict format
        pixel_values = batch['pixel_values']
        pixel_mask = batch['pixel_mask']
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in batch['labels']]
        # Forward pass for loss
        outputs = self.model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            labels=targets
        )
        
        loss = outputs.loss
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        
        # Get predictions for metrics
        with torch.no_grad():
            predictions = self.model(
                pixel_values=pixel_values
            )
        
        # Format predictions for metrics
        pred_boxes = predictions.pred_boxes
        pred_logits = predictions.logits
        
        # Process predictions for metrics
        metric_preds = []
        for i in range(len(targets)):
            
            # Get predictions for this image
            probs = torch.softmax(pred_logits[i], dim=-1)
            
            # Handle background class
            background_probs = probs[:, 0]
            class_probs, labels = torch.max(probs[:, 1:], dim=-1)
            labels = labels + 1  # Offset by 1 since we excluded background
            
            # Filter out background predictions
            is_foreground = class_probs > background_probs
            boxes = pred_boxes[i][is_foreground]
            scores = class_probs[is_foreground]
            labels = labels[is_foreground]
            
            metric_preds.append({
                "boxes": boxes,
                "scores": scores,
                "labels": labels,
            })
            metric_targets = []
            for target in targets:
                # print("Target: ", target)
                metric_targets.append({
                    "boxes": target["boxes"],
                    "labels": target["class_labels"],
                })
        
        # Update metrics
        self.metric.update(preds=metric_preds, target=metric_targets)
        
        return loss
    
    def on_validation_epoch_end(self):
        # Compute and log detection metrics
        metrics = self.metric.compute()
        
        # Log map and map_50
        self.log("val_map", metrics["map"], on_epoch=True)
        self.log("val_map_50", metrics["map_50"], on_epoch=True)
        self.log("val_map_75", metrics["map_75"], on_epoch=True)
        
        # Reset metrics
        self.metric.reset()
    
    def configure_optimizers(self):
        # Separate backbone and detection head parameters for different learning rates
        backbone_params = []
        head_params = []
        
        for name, param in self.model.named_parameters():
            if "backbone" in name:
                backbone_params.append(param)
            else:
                head_params.append(param)
        
        # Create optimizer with different learning rates
        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': self.learning_rate * 0.1},
            {'params': head_params, 'lr': self.learning_rate}
        ], weight_decay=self.weight_decay)
        
        # Learning rate scheduler
        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=[self.learning_rate * 0.1, self.learning_rate],
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.1,  # Warmup for 10% of training
                div_factor=25,
                final_div_factor=1000,
            ),
            "interval": "step",
            "frequency": 1
        }
        
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
 