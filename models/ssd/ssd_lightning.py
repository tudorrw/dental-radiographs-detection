import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import batched_nms
from models.ssd.utils.ssd import SSD


class SSDLightning(pl.LightningModule):
    def __init__(self, num_classes, learning_rate):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.model = SSD(num_classes=num_classes)
        self.metric = MeanAveragePrecision(box_format="xyxy")

    def forward(self, images):
        images = torch.stack(images, dim=0)
        return self.model(images)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        images = torch.stack(images, dim=0)
        head_outputs, anchors = self.model(images)
        loss_dict = self.compute_loss(head_outputs, targets, anchors)
        total_loss = sum(loss for loss in loss_dict.values())
        self.log_dict(loss_dict, prog_bar=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        images = torch.stack(images, dim=0)

        # Forward pass
        head_outputs, anchors = self.model(images)

        # Post-process predictions
        preds = self.postprocess_detections(head_outputs, anchors)

        # Update metric with correctly formatted predictions
        self.metric.update(preds, targets)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.hparams.learning_rate)

    def postprocess_detections(self, head_outputs, anchors, score_thresh=0.5, nms_thresh=0.5):
        """Convert model outputs into a format compatible with MeanAveragePrecision metric."""
        bbox_regression = head_outputs["bbox_regression"]
        cls_logits = nn.Softmax(head_outputs["cls_logits"], dim=-1)  # Convert logits to probabilities
        num_classes = cls_logits.size(-1)

        preds = []
        for boxes, scores, anchors_per_image in zip(bbox_regression, cls_logits, anchors):
            boxes = self.model.box_coder.decode_single(boxes, anchors_per_image)

            image_preds = {"boxes": [], "scores": [], "labels": []}
            for label in range(1, num_classes):  # Skip background class (label 0)
                score = scores[:, label]
                keep_idxs = score > score_thresh
                score = score[keep_idxs]
                box = boxes[keep_idxs]

                # Keep top scoring predictions before NMS
                num_topk = min(len(score), 400)
                score, idxs = score.topk(num_topk)
                box = box[idxs]

                # Apply Non-Maximum Suppression (NMS)
                keep = batched_nms(box, score, torch.full_like(score, label, dtype=torch.int64), nms_thresh)

                image_preds["boxes"].append(box[keep])
                image_preds["scores"].append(score[keep])
                image_preds["labels"].append(torch.full_like(score[keep], label, dtype=torch.int64))

            # Convert lists to tensors
            if image_preds["boxes"]:
                image_preds["boxes"] = torch.cat(image_preds["boxes"], dim=0)
                image_preds["scores"] = torch.cat(image_preds["scores"], dim=0)
                image_preds["labels"] = torch.cat(image_preds["labels"], dim=0)
            else:
                # No detections, create empty tensors
                image_preds["boxes"] = torch.empty((0, 4), device=boxes.device)
                image_preds["scores"] = torch.empty((0,), device=boxes.device)
                image_preds["labels"] = torch.empty((0,), dtype=torch.int64, device=boxes.device)

            preds.append(image_preds)

        return preds
