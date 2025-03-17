import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import yaml
import pytorch_lightning as pl
from models.ssd.simplified_ssd_lightning import SimplifiedSSDLightning
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader


from data.panoramic_dataset import PanoramicDataset

if __name__ == "__main__":

# Navigate to the "faster_rcnn" folder where cfg.yaml is located
    CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cfg.yaml")

    print("Starting training...")
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)
    print("Config:", config)

    # Load datasets
    train_dataset = PanoramicDataset(
        csv_path=f"{config['image_dir']}/{config['data_type']}/quadrant_enumeration_train.csv",
        image_dir=f"{config['image_dir']}/{config['data_type']}/xrays", 
        dataset_type="train"
    )
    val_dataset = PanoramicDataset(
        csv_path=f"{config['image_dir']}/{config['data_type']}/quadrant_enumeration_val.csv",
        image_dir=f"{config['image_dir']}/{config['data_type']}/xrays",
        dataset_type="val"
    )


    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], num_workers=4, collate_fn=PanoramicDataset.collate_fn)

    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], num_workers=4, collate_fn=PanoramicDataset.collate_fn)

    # Model
    model = SimplifiedSSDLightning(num_classes=config["n_teeth"] + 1, learning_rate=float(config["learning_rate"]), momentum=float(config["momentum"]))

    # Logger & Checkpoints
    logger = TensorBoardLogger(save_dir=config["checkpoints_path"], name="ssd")

    #create version-checkpoint directory
    version_dir = os.path.join(config["checkpoints_path"], logger.name, f"version_{logger.version}")
    os.makedirs(version_dir, exist_ok=True)
    
    # save_top_k=1: Only the best model (in terms of the lowest training loss) will be saved.
    # mode='min': The checkpoint is saved when the monitored value (train_loss) decreases.
    # dirpath='checkpoints/': Specifies the directory where the checkpoint is saved.
    checkpoint_callback = ModelCheckpoint(
        dirpath=version_dir,
        filename="best-checkpoint-{epoch:02d}-{val_loss:.2f}", 
        monitor="val_loss", 
        mode="min", 
        save_top_k=1,
        save_last=True
    )
    trainer = pl.Trainer(
        max_epochs=config["max_epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback],
        val_check_interval=1.0,
        num_sanity_val_steps=2
    )

    # Train
    trainer.fit(model, train_loader, val_loader)
    print("Training ended...")
