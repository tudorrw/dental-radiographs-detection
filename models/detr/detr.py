import torch
import numpy as np
import pytorch_lightning as pl
from transformers import DetrForObjectDetection, DetrImageProcessor
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
 
class DETR(pl.LightningModule):
    def __init__(self, num_classes, learning_rate=1e-5, weight_decay=1e-4, use_weighted_loss=True):
        super().__init__()
       
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.use_weighted_loss = use_weighted_loss
       
        # Load pre-trained DETR model
        self.model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50",
            num_labels=self.num_classes,
            ignore_mismatched_sizes=True
        )
       
        # Metrics
        self.metric = MeanAveragePrecision(box_format="xywh")
        
        # For confusion matrix
        self.val_pred_labels = []
        self.val_true_labels = []
        
        # Create id2label mapping for class names
        self.id2label = {i: f"Class {i}" for i in range(self.num_classes+1)}  # +1 for background

        self.class_weights = torch.ones(self.num_classes + 1, dtype=torch.float32)
        self.class_weights[1:17] = 2.0
       
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

        # increase loss for the upper teeth - has an improvement to the conf matrix
        # if self.training and self.use_weighted_loss:
        #     for label_dict in labels:
        #         if 'class_labels' in label_dict:
        #             class_labels = label_dict['class_labels']
        #             #gives more weight to samples containing upper teeth, i think all images
        #             upper_teeth_mask = (class_labels >= 1) & (class_labels <= 16)
        #             if upper_teeth_mask.any():
        #                 loss = loss * 1.5
        #                 break
            

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
        metric_targets = []
        
        for i in range(len(targets)):
           
            # Get predictions for this image
            probs = torch.softmax(pred_logits[i], dim=-1)
           
            # Handle background class
            background_probs = probs[:, 0]
            class_probs, labels = torch.max(probs[:, 1:], dim=-1)
            labels = labels + 1  # Offset by 1 since we excluded background
           
            # Filter out background predictions
            threshold = 0.7  # Confidence threshold
            is_foreground = class_probs > threshold
            boxes = pred_boxes[i][is_foreground]
            scores = class_probs[is_foreground]
            pred_labels = labels[is_foreground]
           
            metric_preds.append({
                "boxes": boxes,
                "scores": scores,
                "labels": pred_labels,
            })
            
            # Prepare target for this image
            target = targets[i]
            metric_targets.append({
                "boxes": target["boxes"],
                "labels": target["class_labels"],
            })
            
            # Collect labels for confusion matrix (using IoU matching)
            # Skip if no predictions or no ground truth
            if len(target["boxes"]) > 0 and len(boxes) > 0:
                # Convert boxes to xyxy format for IoU calculation
                pred_boxes_xyxy = self._xywh_to_xyxy(boxes)
                gt_boxes_xyxy = self._xywh_to_xyxy(target["boxes"])
                
                # Compute IoU between each pred and gt box
                ious = self._box_iou(pred_boxes_xyxy, gt_boxes_xyxy)
                
                # For each ground truth box, find the best matching prediction
                for gt_idx in range(len(target["class_labels"])):
                    gt_label = target["class_labels"][gt_idx].item()
                    self.val_true_labels.append(gt_label)
                    
                    if len(ious) == 0:  # No predictions
                        self.val_pred_labels.append(0)  # Background
                        continue
                    
                    # Find best prediction match
                    best_iou, best_idx = torch.max(ious[:, gt_idx], dim=0)
                    
                    # If IoU is high enough, consider it a match
                    if best_iou > 0.5:
                        self.val_pred_labels.append(pred_labels[best_idx].item())
                        
                        # Remove this prediction to avoid double matching
                        mask = torch.ones(ious.size(0), dtype=torch.bool)
                        mask[best_idx] = False
                        ious = ious[mask]
                        pred_labels = pred_labels[mask]
                    else:
                        # No match, count as background prediction
                        self.val_pred_labels.append(0)
 
      
        # Update metrics
        self.metric.update(preds=metric_preds, target=metric_targets)
       
        return loss
   
    def on_validation_epoch_start(self):
        # Reset metrics and confusion matrix data
        self.metric.reset()
        self.val_pred_labels = []
        self.val_true_labels = []
        
    def on_validation_epoch_end(self):
        # Compute and log detection metrics
        metrics = self.metric.compute()
       
        # Log map and map_50
        self.log("val_map", metrics["map"], on_epoch=True)
        self.log("val_map_50", metrics["map_50"], on_epoch=True)
        self.log("val_map_75", metrics["map_75"], on_epoch=True)
        
        # Compute and log confusion matrix if we have predictions
        if self.val_pred_labels and self.val_true_labels:
            self.compute_confusion_matrix()
       
        # Reset metrics
        self.metric.reset()
   
    def _xywh_to_xyxy(self, boxes):
        """Convert boxes from (x_center, y_center, width, height) to (x1, y1, x2, y2)."""
        x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        return torch.stack([x1, y1, x2, y2], dim=1)
    
    def _box_iou(self, boxes1, boxes2):
        """
        Compute IoU between two sets of boxes.
        Boxes are in (x1, y1, x2, y2) format.
        """
        # Calculate intersection
        max_xy = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
        min_xy = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
        inter = torch.clamp((max_xy - min_xy), min=0)
        inter = inter[:, :, 0] * inter[:, :, 1]
        
        # Calculate union
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        union = area1[:, None] + area2[None, :] - inter
        
        # Calculate IoU
        iou = inter / union
        return iou
    
    def compute_confusion_matrix(self):
        """Compute and visualize confusion matrix from collected predictions and targets."""
        if not self.val_pred_labels or not self.val_true_labels:
            print("No validation predictions collected for confusion matrix")
            return
        
        # Convert lists to numpy arrays
        y_true = np.array(self.val_true_labels)
        y_pred = np.array(self.val_pred_labels)
        
        # Get classes
        classes = list(range(self.num_classes + 1))  # Including background (0)

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        
        # Plot confusion matrix
        plt.figure(figsize=(16, 14))
        
        # Normalize confusion matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)  # Replace NaN with 0
        
        # Create heatmap with class names
        sns.heatmap(
            cm_norm,
            annot=True,
            cmap="Blues",
            fmt='.2f',
            square=True,
            xticklabels=[self.id2label.get(i, f"Class {i}") for i in classes],
            yticklabels=[self.id2label.get(i, f"Class {i}") for i in classes]
        )
        
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Normalized Confusion Matrix')
        
        # Save figure to logger directory if available
        if self.logger and hasattr(self.logger, 'log_dir'):
            save_dir = self.logger.log_dir
            epoch = self.current_epoch
            plt.savefig(os.path.join(save_dir, f"confusion_matrix_epoch_{epoch}.png"))
            
        plt.close('all')
        
        # Calculate and log per-class metrics
        precision = np.diag(cm) / np.sum(cm, axis=0)
        recall = np.diag(cm) / np.sum(cm, axis=1)
        f1_score = 2 * precision * recall / (precision + recall)
        
        # Replace NaN with 0
        precision = np.nan_to_num(precision)
        recall = np.nan_to_num(recall)
        f1_score = np.nan_to_num(f1_score)
        
        # Log metrics
        # for i in range(1, len(classes)):  # Skip background class
        #     tooth_name = self.id2label.get(i, f"Class_{i}")
        #     self.log(f"val_precision_{tooth_name}", precision[i], on_epoch=True)
        #     self.log(f"val_recall_{tooth_name}", recall[i], on_epoch=True)
        #     self.log(f"val_f1_{tooth_name}", f1_score[i], on_epoch=True)
        
        # Log average metrics (excluding background)
        self.log("val_precision_avg", np.mean(precision[1:]), on_epoch=True)
        self.log("val_recall_avg", np.mean(recall[1:]), on_epoch=True)
        self.log("val_f1_avg", np.mean(f1_score[1:]), on_epoch=True)
        
        # Reset for next epoch
        self.val_pred_labels = []
        self.val_true_labels = []
    
    def configure_optimizers(self):
        # Separate backbone and detection head parameters for different learning rates
        backbone_params = [] # Parameters from the backbone (ResNet50)
        head_params = []    # Parameters from the detection head and trandformers
       
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
 