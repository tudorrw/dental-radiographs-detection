import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import yaml
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from data.rcnn_teeth_dataset import TeethDataset
from utils.mapper import ToothLabelMapper
from tqdm import tqdm

def visualize_ground_truth(dataset, idx, save_path=None, figsize=(16, 10)):
    label_mapper = ToothLabelMapper()
   
    image, targets = dataset[idx]["image"], dataset[idx]["targets"]  # Get image and annotations
    image_np = image.squeeze().numpy()  # Remove extra dimensions and convert to numpy
    image_np = np.clip(image_np, 0, 1)  # Ensure values are between 0 and 1
    
    # Convert to PIL image for display (single channel)
    image_pil = Image.fromarray((image_np * 255).astype(np.uint8), mode='L')
    
    # Get bounding boxes and labels
    boxes = targets['boxes'].numpy()  # Pascal VOC format [x1, y1, x2, y2]
    class_ids = targets['labels'].numpy()

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image_pil, cmap='gray')  # Use grayscale colormap
    
    # Map class indices back to tooth numbers for more meaningful labels
    tooth_numbers = label_mapper.decode(class_ids)
    
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
        plt.close(fig)  # Close the figure to free memory
        return True
    
    return fig

def main():
    CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cfg.yaml")

    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    os.makedirs("visualizations", exist_ok=True)

    # Load dataset
    dataset = TeethDataset(
        image_dir=f"{config['image_dir']}/{config['data_type']}/xrays", 
        csv_path=f"{config['csv_path']}/{config['data_type']}/quadrant_enumeration_voc_val.csv",
    )
    
    # Process all images
    print(f"Processing {len(dataset)} images...")
    for idx in tqdm(range(len(dataset))):
        image = dataset.data.iloc[idx]
        file_name = image["file_name"].split(".")[0]

        save_path = os.path.join("visualizations", f"sample_{file_name}.png")
        visualize_ground_truth(dataset, idx, save_path=save_path)
    
    print("All visualizations have been saved to folder 'visualizations'")

if __name__ == "__main__":
    main()
