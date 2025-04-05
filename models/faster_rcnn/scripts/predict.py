import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import yaml
import pytorch_lightning as pl
from models.faster_rcnn.faster_rcnn import FasterRCNN
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping 
from torch.utils.data import DataLoader


from data.rcnn_teeth_dataset import TeethDataset

if __name__ == "__main__":

# Navigate to the "faster_rcnn" folder where cfg.yaml is located
    CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cfg.yaml")

    print("Starting predicting...")
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)
    print("Config:", config)

        # Load datasets
    predict_dataset = TeethDataset(
        image_dir=f"{config['image_dir']}/{config['data_type']}/xrays", 
        csv_path=f"{config['csv_path']}/{config['data_type']}/quadrant_enumeration_voc_test.csv",
    )

    predict_loader = DataLoader(predict_dataset, batch_size=config["batch_size"], num_workers=0, pin_memory=True, collate_fn=TeethDataset.collate_fn)
    checkpoint_dict = dict(version="version_0", filename="epoch=44-val_loss=0.94.ckpt")
    checkpoint = os.path.join(config["checkpoints_path"], "faster_rcnn", checkpoint_dict["version"], checkpoint_dict["filename"])
    
    model = FasterRCNN.load_from_checkpoint(checkpoint)
    model.hparams.output_dir = f"{config['output_dir']}/{checkpoint_dict['version']}__{checkpoint_dict['filename'].rsplit('.', 1)[0]}"
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
    )
    trainer.predict(model, predict_loader)
