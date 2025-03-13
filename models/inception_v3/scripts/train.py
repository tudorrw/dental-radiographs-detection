import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import yaml
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from data.teeth_dataset import TeethDataset
from models.inception_v3.inceptionv3 import InceptionV3Lightning

if __name__ == "__main__":
    # Load configuration file
    CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cfg.yaml")

    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load datasets
    train_dataset = TeethDataset(
        csv_path=f"{config['image_dir']}/{config['data_type']}/quadrant_enumeration_train.csv",
        image_dir=f"{config['image_dir']}/{config['data_type']}/xrays",
        dataset_type="train"
    )
    val_dataset = TeethDataset(
        csv_path=f"{config['image_dir']}/{config['data_type']}/quadrant_enumeration_val.csv",
        image_dir=f"{config['image_dir']}/{config['data_type']}/xrays",
        dataset_type="val"
    )

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], num_workers=4, collate_fn=TeethDataset.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], num_workers=4, collate_fn=TeethDataset.collate_fn)


    # for batch in train_loader:
    #     images, targets = batch
    #     print(f"Images shape: {images.shape}")
    #     print(f"Targets: {targets}")
    #     break  # Only check the first batch


    # Initialize model
    model = InceptionV3Lightning(num_classes=config["n_teeth"], learning_rate=float(config["learning_rate"]))

    # Set up logging and checkpoints
    logger = TensorBoardLogger(save_dir=config["checkpoints_path"], name="google_net")
    checkpoint_callback = ModelCheckpoint(
        dirpath=config["checkpoints_path"], monitor="val_loss", mode="min", save_top_k=1
    )

    # Trainer setup
    trainer = pl.Trainer(
        max_epochs=config["max_epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback],
        check_val_every_n_epoch=5
    )

    # Start Training
    trainer.fit(model, train_loader, val_loader)
    print("Training completed successfully!")