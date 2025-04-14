import os
import yaml
from models.faster_rcnn.faster_rcnn import FasterRCNN
from models.retinanet.retinanet import RetinaNet
from ultralytics import YOLO

class FineTunedModels:
    def __init__(self, device):
        self.device = device
        
    def get_faster_rcnn_model(self):
    # 1) Load config
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models",
            "faster_rcnn",
            "cfg.yaml"
        )
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # 2) Load model checkpoint
        # checkpoint_dict = dict(version="version_0", filename="epoch=44-val_loss=0.94.ckpt")
        checkpoint_dict = dict(version="version_2", filename="epoch=27-val_loss=0.85.ckpt")
        checkpoint = os.path.join(
            config["checkpoints_path"],
            "faster_rcnn",
            checkpoint_dict["version"],
            checkpoint_dict["filename"]
        )
        model = FasterRCNN.load_from_checkpoint(checkpoint)
        model.eval()
        model.to(self.device)
        return model
    
    def get_yolo_model(self):
        checkpoint = os.path.join("runs", "detect", "train", "weights", "best.pt")
        model = YOLO(checkpoint)
        return model
    
    def get_retinanet_model(self):
    # 1) Load config
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models",
            "retinanet",
            "cfg.yaml"
        )
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # 2) Load model checkpoint
        checkpoint_dict = dict(version="version_2", filename="epoch=11-val_loss=0.45.ckpt")
        # checkpoint_dict = dict(version="version_1", filename="epoch=19-val_loss=0.90.ckpt")
        checkpoint = os.path.join(
            config["checkpoints_path"],
            "retinanet",
            checkpoint_dict["version"],
            checkpoint_dict["filename"]
        )
        model = RetinaNet.load_from_checkpoint(checkpoint)
        model.eval()
        model.to(self.device)
        return model
