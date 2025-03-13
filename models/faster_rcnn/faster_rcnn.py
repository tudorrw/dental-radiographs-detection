import torch
import pytorch_lightning as L
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

import matplotlib.pyplot as plt
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics.classification import MulticlassConfusionMatrix
import seaborn as sns

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
    def __init__(self, num_classes, learning_rate, momentum):
        super(FasterRCNN, self).__init__()

        self.save_hyperparameters()
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.momentum = momentum
        
        self.model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=self.num_classes)

        self.metric = MeanAveragePrecision(box_format="xyxy")
        self.confmatrix = MulticlassConfusionMatrix(num_classes=self.num_classes)


    def forward(self, images, targets=None):
        return self.model(images, targets)

    def training_step(self, batch, batch_idx):
        images, targets = batch

        targets=[{k: v for k, v in t.items()} for t in targets]

        loss_dict = self.forward(images, targets)

        total_loss = sum(loss for loss in loss_dict.values())
        self.log("train_loss", total_loss)
    
        return total_loss
    
    

    # Trainer adds torch.no_grad() for the validation loop, so anyrhing in the validation_step() method will be already with gradients disabled
    def validation_step(self, batch, batch_idx):
        images, targets = batch
        # print("Targets: ", targets)
        # print("Images: ", images)

        # Set model in training mode to get access to losses
        self.model.train() # imputs ran
        loss_dict = self.forward(images, targets)
        total_loss = sum(loss for loss in loss_dict.values())
        self.model.eval()
        predictions = self.model(images)
        # print("Targets: ", targets)
        # print("Predictions: ", predictions)
        # Set model in eval mode to obtain predictions
        pred_labels = torch.cat([pred["labels"] for pred in predictions], dim=0)
        true_labels = torch.cat([t["labels"] for t in targets], dim=0)
        # self.confmatrix(preds=pred_labels, target=true_labels)
        # self.confmatrix.update(preds=predictions, target=targets)
        # print("Pred labeld: ", pred_labels, "true labels: ", true_labels)

        self.metric.update(preds=predictions, target=targets)
        self.log("val_loss", total_loss)

        return total_loss
    
    def on_validation_epoch_end(self):
        computed_metrics = self.metric.compute()
    
        # Filter out keys that start with "mar_" and the "classes" key
        filtered_metrics = {k: v for k, v in computed_metrics.items() if not k.startswith("mar_") and not k == "classes"}
        for k, v in filtered_metrics.items():
            self.log(f"val_{k}", v)

        self.metric.reset()
        # fig, ax = self.plot_confusion_matrix()
        # self.logger.experiment.add_figure("Confusion Matrix", fig, self.current_epoch)


    # def plot_confusion_matrix(self):
    #     """Generate confusion matrix plot."""
    #     fig, ax = plt.subplots(figsize=(10, 8))
    #     cm = self.confmatrix.compute().cpu().numpy()
    #     sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    #     ax.set_xlabel("Predicted Labels")
    #     ax.set_ylabel("True Labels")
    #     ax.set_title("Confusion Matrix")
    #     plt.close(fig)
    #     return fig, ax

    def configure_optimizers(self):
        return torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum)