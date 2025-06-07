import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import torch
import yaml
from utils.mapper import ToothLabelMapper
from data.detr_teeth_dataset import CocoDetectionTeeth
from transformers import DetrImageProcessor
import albumentations as A


def get_augmentations(): 
    return A.Compose([
        # A.RandomBrightnessContrast(p=.5),
        # A.ShiftScaleRotate(p=.5,
        #                 shift_limit=0.05,
        #                 rotate_limit=10),
        # A.CoarseDropout(num_holes_range=(5,5),
        #         hole_height_range=(70,80),
        #         hole_width_range=(70,80),
        #         fill=128,
        #         p=1.0),
            A.Downscale(
                scale_range=[0.25, 0.25],
                interpolation_pair={"upscale":0,"downscale":0},
                p=1.0
            ),
            A.HEStain(
                method="random_preset",
                intensity_scale_range=[0.7, 1.3],
                intensity_shift_range=[-0.2, 0.2],
                augment_background=False,   
                p=1.0
            )
        # A.CLAHE(clip_limit=2.0, tile_grid_size=(16,16), p=0.5),
    ],
    bbox_params=A.BboxParams(format='coco',label_fields=["category_ids"], clip=True)
    )

def visualize_ground_truth(dataset, idx, save_path=None, figsize=(16, 10), transform=None):
    """
    Visualize a dataset sample with ground truth annotations.
    
    Args:
        dataset: CocoDetectionTeeth dataset
        idx: Index of the sample to visualize
        save_path: Path to save the visualization (if None, just display)
        figsize: Figure size
    """
    # Initialize tooth label mapper
    label_mapper = ToothLabelMapper()

    image_id = dataset.ids[idx]    
    # Get image and annotations

    img_path = dataset.coco.loadImgs(image_id)[0]['file_name']
    image = Image.open(os.path.join(dataset.root, img_path))
    targets = dataset.coco.loadAnns(dataset.coco.getAnnIds(imgIds=image_id))
    
    # Convert PIL image to numpy array
    img_np = np.array(image)
    
    # Set colors for the bounding boxes
    bbox_color = 'darkgreen'
    text_color = 'white'

    bboxes, category_ids = dataset.encode_targets(targets)

    if transform:
        augm = get_augmentations()
        augmented = augm(image=img_np, bboxes=bboxes, category_ids=category_ids)
        image = augmented["image"]
        bboxes = augmented["bboxes"]
        category_ids = [int(c) for c in augmented["category_ids"]] 

        # Update transformed annotations
        for i, target in enumerate(targets):
            target["bbox"] = bboxes[i]
            target["category_id"] = category_ids[i]
        image = Image.fromarray(image)   

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Display the image
    ax.imshow(image)
    # Draw bounding boxes and labels
    for target in targets:
        
        # Get bbox coordinates (COCO format: [x, y, width, height])
        x, y, width, height = target["bbox"]
        class_idx = target["category_id"]
        
        # Create rectangle patch
        rect = patches.Rectangle(
            (x, y), width, height,
            linewidth=2, edgecolor=bbox_color, facecolor='none'
        )
        
        # Add rectangle to the plot
        ax.add_patch(rect)
        
        # Add label with tooth number and class index
        # label_text = f"{tooth_id} (Class {class_idx})"
        label_text = class_idx
        ax.text(
            x, y - 5, label_text,
            color=text_color, fontsize=10,
            bbox=dict(facecolor=bbox_color, alpha=0.7, pad=1)
        )
    
    # Add title with image info
    ax.set_title(f"Image ID: {image_id}, {len(targets)} teeth annotations")
    
    # Remove axis
    ax.axis('off')
    
    # Make sure everything fits
    plt.tight_layout()
    
    # Save or show the visualization
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    return fig
 

def main():
    # Fixed configurations
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cfg.yaml")
    num_samples = 5
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "visualizations")
    seed = 42
    
    # Set random seed
    random.seed(seed)
    
    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Initialize the image processor
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    
    # Load train dataset
    train_dataset = CocoDetectionTeeth(
        json_path=f"{config['csv_path']}/{config['data_type']}/{config['data_type']}_coco_train.json",
        image_dir=f"{config['image_dir']}/{config['data_type']}/xrays",
        processor=processor,
        train_mode=False  # No augmentation for visualization
    )
    
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Choose random indices to visualize
    indices = random.sample(range(len(train_dataset)), min(num_samples, len(train_dataset)))
    
    # Visualize each sample
    for i, idx in enumerate(indices):
        save_path = os.path.join(save_dir, f"sample_{i+1}_idx_{idx}.png")
        # visualize_ground_truth(train_dataset, idx, save_path=save_path)
        visualize_ground_truth(train_dataset, idx, save_path=save_path, transform=True)
 
if __name__ == "__main__":
    main()
