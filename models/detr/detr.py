import torch
import numpy as np
import pytorch_lightning as pl
from transformers import DetrForObjectDetection, DeformableDetrForObjectDetection
from torchvision.ops import box_iou

from torchmetrics.detection.mean_ap import MeanAveragePrecision
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
 
class DETR(pl.LightningModule):
    def __init__(self, num_classes, learning_rate, weight_decay, use_weighted_loss=True):
        super().__init__()
       
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.use_weighted_loss = use_weighted_loss
       
        # Load pre-trained DETR model
        # self.model = DetrForObjectDetection.from_pretrained(
        #     "facebook/detr-resnet-50-dc5",
        #     num_labels=self.num_classes,
        #     ignore_mismatched_sizes=True
        # )
        self.model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50",
            num_labels=self.num_classes,
            ignore_mismatched_sizes=True
        )

        self.id2label = {i: f"Class {i}" for i in range(self.num_classes+1)}  # +1 for background

        # Metrics
        self.metric = MeanAveragePrecision(box_format="xywh")
        
        # For confusion matrix
        self.val_pred_labels = []
        self.val_true_labels = []
       
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
        return loss
 
    def training_step(self, batch, batch_idx):
        # Extract from dict format
        loss = self.common_step(batch, batch_idx)
        self.log("train_loss", loss)
        # for k,v in loss_dict.items():
        #     self.log(f"train_{k}", v)
        return loss
   
    def validation_step(self, batch, batch_idx):
        pixel_values = batch['pixel_values']
        pixel_mask = batch['pixel_mask']
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in batch['labels']]

        # Set model in training mode to get loss
        self.model.train()
        outputs = self.forward(pixel_values, pixel_mask, labels=targets)
        val_loss = outputs.loss

        self.model.eval()
        with torch.no_grad():
            preds = self.model(pixel_values)

        pred_boxes = preds.pred_boxes
        pred_logits = preds.logits

        metric_preds = []
        metric_targets = []

        for i in range(len(targets)):
            probs = torch.softmax(pred_logits[i], dim=-1)
            background_probs = probs[:, 0]
            class_probs, labels = torch.max(probs[:, 1:], dim=-1)
            labels = labels + 1  # Shift because background is 0

            threshold = 0.5
            keep = class_probs > threshold
            boxes = pred_boxes[i][keep]
            scores = class_probs[keep]
            pred_labels = labels[keep]

            metric_preds.append({
                "boxes": boxes,
                "scores": scores,
                "labels": pred_labels,
            })

            metric_targets.append({
                "boxes": targets[i]["boxes"],
                "labels": targets[i]["class_labels"],
            })

            # For confusion matrix
            if len(targets[i]["boxes"]) > 0 and len(boxes) > 0:
                pred_boxes_xyxy = self._xywh_to_xyxy(boxes)
                gt_boxes_xyxy = self._xywh_to_xyxy(targets[i]["boxes"])

                ious = box_iou(pred_boxes_xyxy, gt_boxes_xyxy)

                for gt_idx in range(len(targets[i]["class_labels"])):
                    gt_label = targets[i]["class_labels"][gt_idx].item()
                    self.val_true_labels.append(gt_label)

                    if len(ious) == 0:
                        self.val_pred_labels.append(0)
                        continue

                    best_iou, best_idx = torch.max(ious[:, gt_idx], dim=0)
                    if best_iou > 0.5:
                        self.val_pred_labels.append(pred_labels[best_idx].item())
                        mask = torch.ones(ious.shape[0], dtype=torch.bool, device=ious.device)
                        mask[best_idx] = False
                        ious = ious[mask]
                        pred_labels = pred_labels[mask]
                    else:
                        self.val_pred_labels.append(0)

        # Update detection metrics
        self.metric.update(preds=metric_preds, target=metric_targets)

        self.log("val_loss", val_loss)
        return val_loss

   
    def on_validation_epoch_start(self):
        # Reset metrics and confusion matrix data
        self.metric.reset()
        self.val_pred_labels = []
        self.val_true_labels = []
        
    def on_validation_epoch_end(self):
        # Compute and log detection metrics
        computed_metrics  = self.metric.compute()
       
        # Log map and map_50
        filtered_metrics = {k: v for k, v in computed_metrics.items() if not k.startswith("mar_") and not k == "classes"}
        for k, v in filtered_metrics.items():
            self.log(f"val_{k}", v)
        
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
    
        
        # Log average metrics (excluding background)
        self.log("val_precision_avg", np.mean(precision[1:]), on_epoch=True)
        self.log("val_recall_avg", np.mean(recall[1:]), on_epoch=True)
        self.log("val_f1_avg", np.mean(f1_score[1:]), on_epoch=True)
        
        # Reset for next epoch
        self.val_pred_labels = []
        self.val_true_labels = []
    


    def configure_optimizers(self):

        # Create optimizer with different learning rates
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
       
        return optimizer
        # # Learning rate scheduler
        # lr_scheduler = {
        #     "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
        #         optimizer,
        #         T_max=10,
        #         eta_min=1e-6,
        #     ),
        # }
       
        # return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
 