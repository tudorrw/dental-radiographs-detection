import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from data.panoramic_dataset import PanoramicDataset
 
def visualize_dataset_samples(dataset, num_samples=5):
    """
    Visualize random samples from the dataset with ground truth boxes.
    
    Args:
        dataset: PanoramicDataset instance
        num_samples: Number of samples to visualize
    """
    # Create a figure with subplots
    fig, axes = plt.subplots(num_samples, 1, figsize=(15, 5 * num_samples))
    
    # If only one sample is requested, make axes iterable
    if num_samples == 1:
        axes = [axes]
    
    # Define a colormap for different teeth classes
    colors = plt.cm.rainbow(np.linspace(0, 1, 33))  # 32 teeth + background
    
    # Get random sample indices
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        # Get the image and targets
        image, targets = dataset[idx]
        
        # Convert from tensor back to numpy for visualization
        image_np = image.permute(1, 2, 0).numpy()
        
        # Denormalize the image
        mean = np.array(dataset.DATA_MEANS)
        std = np.array(dataset.DATA_STD)
        image_np = std * image_np + mean
        image_np = np.clip(image_np, 0, 1)
        
        # Display the image
        axes[i].imshow(image_np)
        
        # Draw bounding boxes
        boxes = targets['boxes'].numpy()
        labels = targets['labels'].numpy()
        
        for box, label in zip(boxes, labels):
            # Create rectangle patch
            x, y, width, height = box[0], box[1], box[2] - box[0], box[3] - box[1]
            rect = patches.Rectangle((x, y), width, height,
                                     linewidth=2, edgecolor=colors[label],
                                     facecolor='none')
            
            # Add rectangle to the plot
            axes[i].add_patch(rect)
            
            # Add label text
            axes[i].text(x, y-5, f"Tooth {label}",
                        color='white', fontsize=10,
                        bbox=dict(facecolor=colors[label], alpha=0.7))
        
        # Add sample index to title
        axes[i].set_title(f"Sample {idx}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig("dataset_visualization.png")
    plt.close()
    print(f"Visualization saved to 'dataset_visualization.png'")
 
if __name__ == "__main__":
    # Load configuration
    data_type = "quadrant_enumeration"
    image_dir = "datasets/coco"
    
    # Create dataset
    dataset = PanoramicDataset(
        csv_path=f"{image_dir}/{data_type}/quadrant_enumeration_train.csv",
        image_dir=f"{image_dir}/{data_type}/xrays",
        dataset_type="train"
    )
    
    # Visualize samples
    visualize_dataset_samples(dataset, num_samples=3)