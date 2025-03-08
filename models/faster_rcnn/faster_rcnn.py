import torch
import pytorch_lightning as L
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
import matplotlib.pyplot as plt
from torchmetrics import Accuracy
import seaborn as sns
from sklearn.metrics import confusion_matrix


class FasterRCNN(L.LightningModule):
    def __init__(self, num_classes, lr):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=num_classes)
        self.lr = lr

        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)

        self.val_preds = []
        self.val_labels = []



    def forward(self, images, targets=None):
        return self.model(images, targets)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self.forward(images, targets)
        total_loss = sum(loss for loss in loss_dict.values())
        self.log("train_loss", total_loss)
        return total_loss
    
    
    def validation_step(self, batch, batch_idx):
        images, targets = batch

        # Set model in training mode to get access to losses
        self.model.train()
        loss_dict = self.forward(images, targets)
        total_loss = sum(loss for loss in loss_dict.values())
        self.model.eval()
        self.log("val_loss", total_loss)

        # for confusion matrix
        # predictions = self.model(images)

        # # Lists to hold aligned predictions and targets
        # y_preds_list = []
        # y_true_list = []

        # for pred, target in zip(predictions, targets):
        #     y_pred = pred["labels"].cpu()
        #     y_true = target["labels"].cpu()

        #     # Ensure the number of predictions per image matches the ground truth
        #     y_preds_list.append(y_pred)
        #     y_true_list.append(y_true)

        # # Only log and store when valid matches exist
        # if y_preds_list and y_true_list:
        #     y_preds = torch.cat(y_preds_list)
        #     y_true = torch.cat(y_true_list)

        #     # Compute accuracy
        #     acc = self.accuracy(y_preds, y_true)
        #     self.log("val_acc", acc, batch_size=len(images))

        #     # Store for confusion matrix
        #     self.val_preds.append(y_preds)
        #     self.val_labels.append(y_true)


        return total_loss
    
    # def on_validation_epoch_end(self):
    #     """Compute confusion matrix at the end of validation"""
    #     if len(self.val_preds) > 0 and len(self.val_labels) > 0:
    #         # Flatten lists
    #         val_preds = torch.cat(self.val_preds).cpu().numpy()
    #         val_labels = torch.cat(self.val_labels).cpu().numpy()

    #         # Compute confusion matrix
    #         cm = confusion_matrix(val_labels, val_preds)

    #         # Plot and log to TensorBoard
    #         fig = self.plot_confusion_matrix(cm)
    #         self.logger.experiment.add_figure("Confusion Matrix", fig, self.current_epoch)

    #         # Reset stored predictions and labels for the next epoch
    #         self.val_preds.clear()
    #         self.val_labels.clear()

    # def plot_confusion_matrix(self, cm):
    #     """Helper function to plot confusion matrix"""
    #     fig, ax = plt.subplots(figsize=(10, 8))
    #     sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    #     ax.set_xlabel("Predicted Labels")
    #     ax.set_ylabel("True Labels")
    #     ax.set_title("Confusion Matrix")
    #     plt.close(fig)
    #     return fig

    

    def configure_optimizers(self):
        return torch.optim.SGD(self.model.parameters(), lr=self.lr)