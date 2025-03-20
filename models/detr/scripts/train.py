import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import yaml
import pytorch_lightning as pl
from models.detr.detr import DETR
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from torch.utils.data import DataLoader
from transformers import DetrImageProcessor
 
from data.detr_teeth_dataset import CocoDetectionTeeth
 
if __name__ == "__main__":
    # Navigate to the "detr" folder where cfg.yaml is located
    CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cfg.yaml")
 
    print("Starting DETR training...")
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    print("Device:", device)
    print("Config:", config)
 
    # Initialize the image processor that will be shared between datasets
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    
    # Load datasets using COCO JSON files
    train_dataset = CocoDetectionTeeth(
        json_path=f"{config['csv_path']}/{config['data_type']}/{config['data_type']}_coco_train.json",
        image_dir=f"{config['image_dir']}/{config['data_type']}/xrays",
        processor=processor,
        train_mode = True
    )
    val_dataset = CocoDetectionTeeth(
        json_path=f"{config['csv_path']}/{config['data_type']}/{config['data_type']}_coco_val.json",
        image_dir=f"{config['image_dir']}/{config['data_type']}/xrays",
        processor=processor,
        train_mode = False
    )

 
    # DataLoaders with persistent workers if GPU available
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        num_workers=4,
        collate_fn=CocoDetectionTeeth.collate_fn,
        shuffle=True,
        persistent_workers=True if torch.cuda.is_available() else False
    )
 
    val_loader = DataLoader(
        
        val_dataset,
        batch_size=config["batch_size"],
        num_workers=4,
        collate_fn=CocoDetectionTeeth.collate_fn,
        persistent_workers=True if torch.cuda.is_available() else False
    )
 
    # Create model with proper number of classes from config
    model = DETR(
        num_classes=config["num_classes"],
        learning_rate=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
        use_weighted_loss=True
    )
 
    # Logger & Checkpoints with versioning
    logger = TensorBoardLogger(save_dir=config["checkpoints_path"], name="detr")
    
    # Create version-specific checkpoint directory
    version_dir = os.path.join(config["checkpoints_path"], "detr", f"version_{logger.version}")
    os.makedirs(version_dir, exist_ok=True)
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=version_dir,
        monitor="val_loss",
        mode="min",
        save_top_k=2,
        save_last=True,
        filename="epoch{epoch:02d}-val_loss{val_loss:.4f}"
    )
    
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,  # Stop after 10 epochs without improvement
        mode="min",
        verbose=True
    )
    
    # Initialize trainer with improved settings
    trainer = pl.Trainer(
        max_epochs=config["max_epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor, early_stopping],
        check_val_every_n_epoch=config.get("validation_interval", 5),
        gradient_clip_val=config.get("gradient_clip_val", 0.1),
        precision="16-mixed" if torch.cuda.is_available() else 32  # Use mixed precision for faster training
    )
 
    # Train the model
    trainer.fit(model, train_loader, val_loader)
    print("DETR training completed.")