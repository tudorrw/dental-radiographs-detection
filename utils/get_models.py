import os
import yaml
from models.faster_rcnn.faster_rcnn import FasterRCNN
from models.retinanet.retinanet import RetinaNet
from ultralytics import YOLO
import torch
import models.dino.datasets.transforms as DT
from utils.slconfig import SLConfig
import numpy as np
from PIL import Image


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
        checkpoint_dict = dict(version="version_2", filename="epoch=75-val_loss=0.77.ckpt")
        # checkpoint_dict = dict(version="version_5", filename="epoch=63-val_loss=0.79.ckpt")
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
        checkpoint = os.path.join("checkpoints", "yolo", "train2", "weights", "best.pt")
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
        # checkpoint_dict = dict(version="version_1", filename="epoch=27-val_loss=0.44.ckpt")
        checkpoint_dict = dict(version="version_2", filename="epoch=14-val_loss=0.40.ckpt")
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
    
    def get_dino_model(self):
        
        dino_model_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models/dino/configs/DINO_4scale.py")
        dino_cfg = SLConfig.fromfile(dino_model_config_path)
        dino_cfg.device = "cuda" if self.device == "cuda" else "cpu"
        model_checkpoint_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints/dino/version_1/checkpoint0027.pth")
        model, criterion, postprocessors = self.build_model_main(dino_cfg)
        
        checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        if self.device == "cuda":
            model.cuda()
        model.eval()
        return model, postprocessors
        
        
    def build_model_main(self, args):
        from models.dino.registry import MODULE_BUILD_FUNCS

        assert args.modelname in MODULE_BUILD_FUNCS._module_dict
        build_func = MODULE_BUILD_FUNCS.get(args.modelname)
        model, criterion, postprocessors = build_func(args)
        return model, criterion, postprocessors


