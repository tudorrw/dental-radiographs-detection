import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import yaml
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from data.rcnn_teeth_dataset import TeethDataset
from utils.mapper import ToothLabelMapper

def visualize_ground_truth(dataset, idx, save_path=None, figsize=(16, 10)):
    label_mapper = ToothLabelMapper()
    print(dataset[idx])
   
    image, targets = dataset[idx]["image"], dataset[idx]["targets"]  # Get image and annotations
    image_np = image.permute(1, 2, 0).numpy() # Convert tensor to numpy image
    image_np = np.clip(image_np, 0, 1) # Denormalize (image was normalized to 0-1 range)
    image_pil = Image.fromarray((image_np * 255).astype(np.uint8)) # Convert to PIL image for display
    
    # Get bounding boxes and labels
    boxes = targets['boxes'].numpy()  # Pascal VOC format [x1, y1, x2, y2]
    class_ids = targets['labels'].numpy()

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image_pil)
    
    # # Map class indices back to tooth numbers for more meaningful labels
    tooth_numbers = label_mapper.decode(class_ids)
    print("Tooth Numbers: ", tooth_numbers)
    quadrant_colors = {
        1: 'red',      # Upper right (teeth 11-18)
        2: 'green',    # Upper left (teeth 21-28)
        3: 'blue',     # Lower right (teeth 41-48)
        4: 'purple'    # Lower left (teeth 31-38)
    }
    
    # Draw bounding boxes
    for i, (box, tooth_number) in enumerate(zip(boxes, tooth_numbers)):
        # Determine quadrant for color
        quadrant = (tooth_number // 10) % 10
        color = quadrant_colors.get(quadrant, 'yellow')
        
        # Create rectangle patch (Pascal VOC format: [x1, y1, x2, y2])
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        
        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=2, edgecolor=color, facecolor='none'
        )
        # Add rectangle to plot
        ax.add_patch(rect)
        
        # Add tooth number label
        ax.text(
            x1, y1 - 5, f"Tooth {tooth_number}",
            color='white', fontsize=9, weight='bold',
            bbox=dict(facecolor=color, alpha=0.7, pad=1)
        )
    
    # Add title with dataset info
    sample_info = dataset.data.iloc[idx]
    ax.set_title(f"File: {sample_info['file_name']} - {len(boxes)} teeth annotations")
    
    # Turn off axis
    ax.axis('off')

    plt.tight_layout()
    # Save or show the figure
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    return fig


def main():
    CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cfg.yaml")

    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    save_dir = os.path.join(config["visualization_dir"] ,"ground_truth")
    num_samples = 5
    seed = 42
    random.seed(seed)

    # Load dataset
    dataset = TeethDataset(
        image_dir=f"{config['image_dir']}/{config['data_type']}/xrays", 
        csv_path=f"{config['csv_path']}/{config['data_type']}/quadrant_enumeration_voc_train.csv",
        dataset_type="train"
    )
    os.makedirs(save_dir, exist_ok=True)
    
    # Choose random indices to visualize
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    # Save paths of generated images
    
    # Visualize each sample
    for i, idx in enumerate(indices):
        save_path = os.path.join(save_dir, f"rcnn_sample_{i+1}_idx_{idx}.png")
        visualize_ground_truth(dataset, idx, save_path=save_path)
        print(dataset)

if __name__ == "__main__":
    main()
