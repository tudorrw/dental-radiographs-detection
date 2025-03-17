import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import yaml
import pytorch_lightning as pl
from models.ssd.simple_ssd_lightning import SimpleSDDLightning
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from torch.utils.data import DataLoader
 
from data.panoramic_dataset import PanoramicDataset
 
if __name__ == "__main__":
    # Load configuration
    CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cfg.yaml")
 
    print("Starting training with simple SSD implementation...")
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: {config}")
 
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
 
    # Create data loaders with persistent workers
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        num_workers=4,
        collate_fn=PanoramicDataset.collate_fn,
        shuffle=True,
        persistent_workers=True if torch.cuda.is_available() else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        num_workers=4,
        collate_fn=PanoramicDataset.collate_fn,
        persistent_workers=True if torch.cuda.is_available() else False
    )
 
    # Initialize the model
    model = SimpleSDDLightning(
        num_classes=config["n_teeth"] + 1,  # Add background class
        learning_rate=float(config["learning_rate"]),
        momentum=float(config["momentum"]),
        weight_decay=5e-4
    )
 
    # Configure logging and checkpointing
    logger = TensorBoardLogger(save_dir=config["checkpoints_path"], name="simple_ssd")
    
    # Create version-specific checkpoint directory
    version_dir = os.path.join(config["checkpoints_path"], "simple_ssd", f"version_{logger.version}")
    os.makedirs(version_dir, exist_ok=True)
    
    # Configure callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=version_dir,
        filename="best-{epoch:02d}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=2,
        save_last=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,  # Stop after 10 epochs without improvement
        mode="min",
        verbose=True
    )
    
    # Initialize trainer with gradient accumulation for stability
    trainer = pl.Trainer(
        max_epochs=config["max_epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor, early_stopping],
        val_check_interval=1.0,  # Check validation after each epoch
        num_sanity_val_steps=2,  # Run validation steps at start
        gradient_clip_val=10.0,  # Prevent gradient explosion
        precision="16-mixed" if torch.cuda.is_available() else 32  # Use mixed precision for faster training
    )
 
    # Train the model
    trainer.fit(model, train_loader, val_loader)
    print("Training completed!")