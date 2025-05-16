import os
import csv 
import torch
import numpy as np
import torch.nn as nn
import seaborn as sns
import pytorch_lightning as L
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.ops import box_iou
from sklearn.metrics import confusion_matrix
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights 


class TwoMLPHeadWithDropout(nn.Module):
    def __init__(self, in_channels, representation_size, dropout_prob):
        super(TwoMLPHeadWithDropout, self).__init__()
        self.dropout_prob = dropout_prob

        self.fc6 = nn.Linear(in_channels, representation_size)
        self.dropout1 = nn.Dropout(self.dropout_prob)
        self.fc7 = nn.Linear(representation_size, representation_size)
        self.dropout2 = nn.Dropout(self.dropout_prob)

    def forward(self, x):
        x = x.flatten(start_dim=1)

        x = F.relu(self.fc6(x))
        x = self.dropout1(x)
        x = F.relu(self.fc7(x))
        x = self.dropout2(x)
        return x

'''
Faster R-CNN processes the image:
 
It generates multiple region proposals (potential teeth locations).
Each proposal is assigned a class label (or "background" if it's not an object).
It makes multiple predictions per image:
 
It detects many bounding boxes, sometimes overlapping.
Each prediction comes with a confidence score and a label.
You only have 32 ground-truth labels:
 
The dataset annotates up to 32 teeth per image.
However, Faster R-CNN is making many more predictions because it's finding multiple candidate teeth.
 
'''
class FasterRCNN(L.LightningModule):
    def __init__(self, num_classes, learning_rate, momentum, output_dir=None):
        super(FasterRCNN, self).__init__()
 
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.dropout_prob = 0.25  # Increased dropout
        self.model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)  # Using V2 version

        # Configure ROI heads
        in_channels = self.model.roi_heads.box_head.fc6.in_features
        representation_size = self.model.roi_heads.box_head.fc6.out_features
        # self.model.roi_heads.box_head = TwoMLPHeadWithDropout(in_channels, representation_size, self.dropout_prob)

        # Configure box predictor
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
        
        # Configure ROI pooling
        # self.model.roi_heads.box_roi_pool.output_size = (7, 7)  # Increased ROI pooling size
        

        # print(f"Model: {self.model}")
        # Detection metrics
        self.metric = MeanAveragePrecision(box_format="xyxy")
        
        # For confusion matrix and classification metrics
        self.val_pred_labels = []
        self.val_true_labels = []
        
        # Simple ID to label mapping for confusion matrix
        self.id2label = {i: str(i) for i in range(self.num_classes)}
        self.id2label[0] = "Background"

    def forward(self, images, targets=None):
        return self.model(images, targets)

    def training_step(self, batch, batch_idx):
        images, targets = batch["image"], batch["targets"]
 
        targets=[{k: v for k, v in t.items()} for t in targets]
 
        loss_dict = self.forward(images, targets)
 
        # Weight the losses
        total_loss = sum(loss for loss in loss_dict.values())
        self.log("train_loss", total_loss)
    
        return total_loss
    
    
 
    # Trainer adds torch.no_grad() for the validation loop, so anyrhing in the validation_step() method will be already with gradients disabled
    def validation_step(self, batch, batch_idx):
        images, targets = batch["image"], batch["targets"]

 
        # Set model in training mode to get access to losses
        self.model.train()
        loss_dict = self.forward(images, targets)
        total_loss = sum(loss for loss in loss_dict.values())
        
        # Set model in eval mode for predictions
        self.model.eval()
        predictions = self.model(images)
        
        # Match predictions to ground truth using IoU for confusion matrix
        for img_idx in range(len(targets)):
            # Get predictions and targets for this image
            pred_boxes = predictions[img_idx]["boxes"]
            pred_scores = predictions[img_idx]["scores"]
            pred_labels = predictions[img_idx]["labels"]
            
            # Get ground truth for this image
            true_boxes = targets[img_idx]["boxes"]
            true_labels = targets[img_idx]["labels"]
            
            # Skip if no ground truth or predictions
            if len(true_boxes) == 0 or len(pred_boxes) == 0:
                continue
                
            # Filter predictions by score threshold
            score_threshold = 0.5
            keep_indices = torch.where(pred_scores > score_threshold)[0]
            if len(keep_indices) == 0:
                continue
                
            pred_boxes = pred_boxes[keep_indices]
            pred_labels = pred_labels[keep_indices]
            
            # Calculate IoU between all pred and gt boxes
            ious = box_iou(pred_boxes, true_boxes)
            
            # For each ground truth, find best matching prediction
            for gt_idx in range(len(true_labels)):
                gt_label = true_labels[gt_idx].item()
                self.val_true_labels.append(gt_label)
                
                if len(ious) == 0:  # No predictions for this image
                    self.val_pred_labels.append(0)  # Background
                    continue
                
                # Find best prediction match
                best_iou, best_idx = torch.max(ious[:, gt_idx], dim=0)
                
                # If IoU is high enough, consider it a match
                if best_iou > 0.5:
                    self.val_pred_labels.append(pred_labels[best_idx].item())
                    
                    # Remove this prediction to avoid double matching
                    mask = torch.ones(ious.shape[0], dtype=torch.bool, device=ious.device)
                    mask[best_idx] = False
                    ious = ious[mask]
                    pred_labels = pred_labels[mask]
                else:
                    # No match with high enough IoU
                    self.val_pred_labels.append(0)  # Consider as background
        
        # Update detection metrics
        self.metric.update(preds=predictions, target=targets)
        self.log("val_loss", total_loss)
 
        
    def on_validation_epoch_start(self):
        # Reset stored predictions and labels
        self.val_pred_labels = []
        self.val_true_labels = []
        self.metric.reset()
        
    def on_validation_epoch_end(self):
        # Compute and log detection metrics (mAP etc.)
        computed_metrics = self.metric.compute()
    
        # Filter out keys that start with "mar_" and the "classes" key
        filtered_metrics = {k: v for k, v in computed_metrics.items() if not k.startswith("mar_") and not k == "classes"}
        for k, v in filtered_metrics.items():
            self.log(f"val_{k}", v)
 
        # Compute confusion matrix and classification metrics if we have predictions
        if self.val_pred_labels and self.val_true_labels:
            self.compute_confusion_matrix()
        
        # Reset metrics for next epoch
        self.metric.reset()
        

    def compute_confusion_matrix(self):
        """Compute and visualize confusion matrix with precision/recall metrics."""
        if not self.val_pred_labels or not self.val_true_labels:
            print("No validation predictions collected for confusion matrix")
            return
        
        # Convert lists to numpy arrays
        y_true = np.array(self.val_true_labels)
        y_pred = np.array(self.val_pred_labels)
        
        # Get unique classes (excluding background for metrics)
        classes = sorted(np.unique(np.concatenate([y_true, y_pred])))
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        
        # Plot confusion matrix
        plt.figure(figsize=(16, 14))
        
        # Normalize confusion matrix for better visualization
        row_sums = cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.divide(cm.astype('float'), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

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
            
        # Close all plots to free memory
        plt.close('all')
            
        
        # Calculate and log precision, recall, and F1 score
        precision = np.diag(cm) / np.sum(cm, axis=0)
        recall = np.divide(np.diag(cm), np.sum(cm, axis=1), out=np.zeros_like(np.diag(cm), dtype=float), where=np.sum(cm, axis=1) != 0)

        f1_score = 2 * precision * recall / (precision + recall)
        
        # Replace NaN with 0
        precision = np.nan_to_num(precision)
        recall = np.nan_to_num(recall)
        f1_score = np.nan_to_num(f1_score)
        
        # Calculate and log average metrics (excluding background)
        self.log("val_precision", np.mean(precision[classes != 0] if 0 in classes else precision), on_epoch=True)
        self.log("val_recall", np.mean(recall[classes != 0] if 0 in classes else recall), on_epoch=True)
        self.log("val_f1", np.mean(f1_score[classes != 0] if 0 in classes else f1_score), on_epoch=True)
 
        # Reset stored predictions
        self.val_pred_labels = []
        self.val_true_labels = []


    def test_step(self, batch, batch_idx):
        images, targets = batch["image"], batch["targets"]

 
        # Set model in eval mode for predictions
        predictions = self.model(images)
        
        # Update detection metrics
        self.metric.update(preds=predictions, target=targets)
 

    def on_test_epoch_end(self):
        # Compute and log detection metrics (mAP etc.)
        computed_metrics = self.metric.compute()
    
        # Filter out keys that start with "mar_" and the "classes" key
        filtered_metrics = {k: v for k, v in computed_metrics.items() if not k.startswith("mar_") and not k == "classes"}
        for k, v in filtered_metrics.items():
            self.log(f"val_{k}", v)
        
        # Reset metrics for next epoch
        self.metric.reset()

    def on_predict_start(self):
        if not os.path.exists(f"{self.hparams.output_dir}"):
             os.makedirs(self.hparams.output_dir, exist_ok=True)

    def predict_step(self, batch, batch_idx):
        images, targets, ids = batch["image"], batch["targets"], batch["id"]
        
        # Get predictions from model
        predictions = self.model(images)
        
        
        # Define CSV path
        csv_path = os.path.join(self.hparams.output_dir, "predictions_results.csv")
        
        # Check if file exists to determine if we need to write the header
        file_exists = os.path.isfile(csv_path)
        
        # Open CSV in append mode
        with open(csv_path, 'a', newline='') as csvfile:
            fieldnames = ['image_id', 'prediction_boxes', 'prediction_scores', 'prediction_labels', 
                        'target_boxes', 'target_labels']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Write header if file doesn't exist
            if not file_exists:
                writer.writeheader()
            
            # Process each image in the batch
            for image_id, target, prediction in zip(ids, targets, predictions):
                # Convert tensors to lists for CSV storage
                pred_boxes = prediction["boxes"].detach().cpu().numpy().tolist()
                pred_scores = prediction["scores"].detach().cpu().numpy().tolist()
                pred_labels = prediction["labels"].detach().cpu().numpy().tolist()
                
                target_boxes = target["boxes"].cpu().numpy().tolist()
                target_labels = target["labels"].cpu().numpy().tolist()
                
                # Write row to CSV
                writer.writerow({
                    'image_id': image_id,
                    'prediction_boxes': str(pred_boxes),
                    'prediction_scores': str(pred_scores),
                    'prediction_labels': str(pred_labels),
                    'target_boxes': str(target_boxes),
                    'target_labels': str(target_labels)
                })
        
    

    def configure_optimizers(self):

        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum)
        return optimizer
        # Use AdamW optimizer with weight decay
        # optimizer = torch.optim.AdamW(
        #     self.parameters(),
        #     lr=self.learning_rate,
        #     weight_decay=0.01,
        #     betas=(0.9, 0.999)
        # )
        
        # # Use OneCycleLR scheduler
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(
        #     optimizer,
        #     max_lr=self.learning_rate,
        #     epochs=self.trainer.max_epochs,
        #     steps_per_epoch=self.trainer.max_epochs,
        #     pct_start=0.3,
        #     anneal_strategy='cos',
        #     div_factor=25.0,
        #     final_div_factor=1000.0
        # )
        
        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": {
        #         "scheduler": scheduler,
        #         "interval": "step",
        #     },
        # }