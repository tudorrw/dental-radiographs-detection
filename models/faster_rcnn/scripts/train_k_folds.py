import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import yaml
import pandas as pd
import pytorch_lightning as pl
from sklearn.model_selection import KFold
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from torch.utils.data import DataLoader
from models.faster_rcnn.faster_rcnn import FasterRCNN
from data.rcnn_teeth_dataset import TeethDataset
 
 
def k_fold_cross_validation(config, n_folds=10):
    """
    Main function to perform k-fold cross-validation.
    
    Args:
        config: Configuration dictionary
        n_folds: Number of folds for cross-validation
    """
    # Set random seed for reproducibility
    random_state = 42
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Use existing dataset split
    train_val_csv_path = f"{config['csv_path']}/{config['data_type']}/quadrant_enumeration_voc_train_val_folds.csv"
    
    # Load CSV data for k-fold
    train_val_df = pd.read_csv(train_val_csv_path)
    

    # Create directory for k-fold results
    kfold_dir = os.path.join(config["checkpoints_path"], "faster_rcnn", "k_fold")
    os.makedirs(kfold_dir, exist_ok=True)
    
    # Create directory for fold CSV files
    fold_csv_dir = os.path.join(config["csv_path"], config["data_type"], "k_fold")
    os.makedirs(fold_csv_dir, exist_ok=True)
    
    # Get unique image IDs for k-fold
    unique_images = train_val_df['id'].unique()
    
    # Initialize k-fold cross-validator
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    # Store results for each fold
    fold_results = {}
    
    # Perform k-fold cross-validation
    for fold, (train_idx, val_idx) in enumerate(kfold.split(unique_images)):
        fold_num = fold + 1
        print(f"\n{'='*80}")
        print(f"FOLD {fold_num}/{n_folds}")
        print(f"{'='*80}")
        
        # Get image IDs for this fold
        train_ids = unique_images[train_idx]
        val_ids = unique_images[val_idx]
        
        # Filter dataframes based on image IDs
        fold_train_df = train_val_df[train_val_df['id'].isin(train_ids)]
        fold_val_df = train_val_df[train_val_df['id'].isin(val_ids)]
        
        # Save fold datasets
        train_csv_path = os.path.join(fold_csv_dir, f"fold_{fold_num}_train.csv")
        val_csv_path = os.path.join(fold_csv_dir, f"fold_{fold_num}_val.csv")
        
        fold_train_df.to_csv(train_csv_path, index=False)
        fold_val_df.to_csv(val_csv_path, index=False)
        
        print(f"Fold {fold_num} data:")
        print(f"  - Training: {len(train_ids)} images, {len(fold_train_df)} annotations")
        print(f"  - Validation: {len(val_ids)} images, {len(fold_val_df)} annotations")
        
        # Create datasets for this fold
        train_dataset = TeethDataset(
            image_dir=f"{config['image_dir']}/{config['data_type']}/xrays",
            csv_path=train_csv_path,
            dataset_type="train"
        )
        
        val_dataset = TeethDataset(
            image_dir=f"{config['image_dir']}/{config['data_type']}/xrays",
            csv_path=val_csv_path,
            dataset_type="val"
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            num_workers=4,
            collate_fn=TeethDataset.collate_fn,
            shuffle=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            num_workers=4,
            collate_fn=TeethDataset.collate_fn
        )
        
        # Initialize model
        model = FasterRCNN(
            num_classes=config["n_teeth"] + 1,
            learning_rate=float(config["learning_rate"]),
            momentum=float(config["momentum"])
        )
        
        # Set up logging and checkpoints
        fold_dir = os.path.join(kfold_dir, f"fold_{fold_num}")
        os.makedirs(fold_dir, exist_ok=True)
        
        logger = TensorBoardLogger(
            save_dir=kfold_dir,              # e.g., checkpoints/faster_rcnn/k_fold
            name=f"fold_{fold_num}",
            version="",  # <== disables the version_0 nesting
        )
        
        checkpoint_callback = ModelCheckpoint(
            dirpath=fold_dir,
            monitor="val_loss",
            mode="max",
            save_top_k=2,
            save_last=True,
            filename="{epoch:02d}-{val_loss:.4f}"
        )
        
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=10,
            mode="max",
        )
        
        # Train model
        trainer = pl.Trainer(
            max_epochs=config["max_epochs"],
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            logger=logger,
            callbacks=[checkpoint_callback, lr_monitor, early_stopping],
        )
        
        print(f"Training fold {fold_num}...")
        trainer.fit(model, train_loader, val_loader)
        
        # Evaluate on validation set
        val_results = trainer.validate(model, val_loader)[0]
        print(f"Validation results for fold {fold_num}:")
        for metric, value in val_results.items():
            print(f"  {metric}: {value:.4f}")
        
        # Store results for this fold
        fold_results[f"fold_{fold_num}"] = val_results
        
        # Clean up to save memory
        del model, train_loader, val_loader
        torch.cuda.empty_cache()
    
    # Summarize results
    
    for fold, results in fold_results.items():
        map_value = results.get("val_map", 0)
        map_50_value = results.get("val_map_50", 0)
        map_75_value = results.get("val_map_75", 0)
        print(f"{fold}: mAP: {map_value:.4f}, mAP@50: {map_50_value:.4f}, mAP@75: {map_75_value:.4f}")

    # Find best fold
    best_fold = max(fold_results.keys(), key=lambda k: fold_results[k].get("val_map", 0))
    best_map = fold_results[best_fold].get("val_map", 0)
    print(f"Best fold: {best_fold} with mAP: {best_map:.4f}")

 
 
if __name__ == "__main__":
    # Load configuration
    CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cfg.yaml")
    
    print(f"Loading config from: {CONFIG_PATH}")
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    
    print("Starting 5-fold cross-validation...")
    k_fold_cross_validation(config, n_folds=5)