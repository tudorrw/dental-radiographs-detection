import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pandas as pd
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.ops import nms
import yaml
 
class UniqueClassNMSProcessor:
    """
    Non-Maximum Suppression processor that ensures each class is assigned to at most one box.
    Adapted from the original NMS implementation in the project.
    """
    def __init__(self, iou_threshold=0.5):
        self.iou_threshold = iou_threshold
 
    def __call__(self, output):
        """
        Process predictions with NMS and ensure each class has at most one box
        
        Args:
            output: dictionary with "boxes", "labels", "scores" keys
            
        Returns:
            output_processed: filtered predictions
        """
        boxes = output["boxes"]
        labels = output["labels"]
        scores = output["scores"]
        
        # Apply NMS
        indices = nms(torch.tensor(boxes), torch.tensor(scores), self.iou_threshold)
        indices = indices.numpy()
        
        # Get filtered results
        boxes_processed = boxes[indices]
        labels_processed = labels[indices]
        scores_processed = scores[indices]
        
        # Keep only highest-scoring box for each class
        best_boxes = {}
        for box, label, score in zip(boxes_processed, labels_processed, scores_processed):
            if label not in best_boxes or score > best_boxes[label][1]:
                best_boxes[label] = (box, score)
        
        # Format results
        output_processed = {
            "boxes": [],
            "labels": [],
            "scores": []
        }
        
        for label, (box, score) in best_boxes.items():
            output_processed["boxes"].append(box)
            output_processed["labels"].append(label)
            output_processed["scores"].append(score)
            
        # Convert to numpy arrays
        output_processed["boxes"] = np.array(output_processed["boxes"])
        output_processed["labels"] = np.array(output_processed["labels"])
        output_processed["scores"] = np.array(output_processed["scores"])
        
        return output_processed
 
def visualize_predictions(image_path, predictions, save_path=None, figsize=(16, 10)):
    """
    Visualize predictions on an image, inspired by the project's existing visualization code.
    
    Args:
        image_path: path to the image file
        predictions: dictionary with "boxes", "labels", "scores" keys
        save_path: where to save the visualization
        figsize: figure dimensions
    """
    # Load the image
    image = Image.open(image_path)
    
    # Extract prediction data
    boxes = predictions["boxes"]
    labels = predictions["labels"]
    scores = predictions["scores"]
    
    # Define colors for different classes (similar to original visualize.py)
    colors = ['red', 'green', 'blue', 'purple', 'orange', 'cyan', 'magenta', 'yellow']
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image, cmap='gray' if image.mode == 'L' else None)
    
    # Draw bounding boxes
    for box, label, score in zip(boxes, labels, scores):
        # Get color based on class
        color = colors[int(label) % len(colors)]
        
        # Extract coordinates (Pascal VOC format: [x1, y1, x2, y2])
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        
        # Create and add rectangle
        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add label and score text
        ax.text(
            x1, y1 - 5, f"Class {label} ({score:.2f})",
            color='white', fontsize=9, weight='bold',
            bbox=dict(facecolor=color, alpha=0.7, pad=1)
        )
    
    # Add title and configure plot
    ax.set_title(f"File: {os.path.basename(image_path)} - {len(boxes)} detections")
    ax.axis('off')
    plt.tight_layout()
    
    # Save or display
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.close(fig)
 


def parse_string_to_array(string_repr):
    """
    Parse string representation of arrays from CSV back to numpy arrays
    """
    # Handle empty case
    if string_repr == '[]' or not string_repr:
        return np.array([])
    
    # Remove brackets, split by commas, and convert to float
    clean_str = string_repr.strip('[]')
    if not clean_str:
        return np.array([])
        
    # Split and convert to appropriate numeric type
    values = [float(x.strip()) for x in clean_str.split(',')]
    return np.array(values)



def parse_boxes_string(boxes_str):
    """
    Parse boxes string from CSV to numpy array of box coordinates
    """
    # Handle empty case
    if boxes_str == '[]' or not boxes_str:
        return np.array([]).reshape(0, 4)
        
    # Clean and extract values
    clean_str = boxes_str.strip('[]')
    if not clean_str:
        return np.array([]).reshape(0, 4)
        
    # Split into individual box strings
    box_strings = clean_str.split('],')
    
    boxes = []
    for box_str in box_strings:
        # Clean up the box string and extract coordinates
        box_str = box_str.strip(' []')
        if not box_str:
            continue
            
        coords = [float(x.strip()) for x in box_str.split(',')]
        boxes.append(coords)
        
    return np.array(boxes)



def visualize_from_csv(score_threshold=0.5):
    """
    Process a CSV file with predictions and visualize them.
    
    Args:
        csv_path: path to CSV with predictions
        image_dir: directory containing the images
        checkpoint_name: name for output directory
        score_threshold: minimum confidence score
    """

   # Create output directory
    CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cfg.yaml")

    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    checkpoint_dict = dict(version="version_0", filename="epoch=44-val_loss=0.94.ckpt")
    output_dir = os.path.join(config["visualization_dir"], f"{checkpoint_dict['version']}__{checkpoint_dict['filename'].rsplit('.', 1)[0]}")
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(config["output_dir"], f"{checkpoint_dict['version']}__{checkpoint_dict['filename'].rsplit('.', 1)[0]}", "predictions_results.csv")
    image_dir=f"{config['image_dir']}/{config['data_type']}/xrays" 
    # Load predictions
    df = pd.read_csv(csv_path)
    
    # Initialize NMS processor
    nms_processor = UniqueClassNMSProcessor(iou_threshold=0.5)
    
    # Get unique image IDs
    image_ids = df['image_id'].unique()
    print(f"Processing {len(image_ids)} images")
    
    # Process each image
    for image_id in image_ids:
        # Get predictions for this image
        row = df[df['image_id'] == image_id].iloc[0]
        print("Images id:", image_id)
        # Parse string representations to arrays
        pred_boxes = parse_boxes_string(row['prediction_boxes'])
        pred_scores = parse_string_to_array(row['prediction_scores'])
        pred_labels = parse_string_to_array(row['prediction_labels']).astype(int)

                # Apply score threshold
        mask = pred_scores >= score_threshold
        pred_boxes = pred_boxes[mask]
        pred_labels = pred_labels[mask]
        pred_scores = pred_scores[mask]

         # Format predictions
        predictions = {
            "boxes": pred_boxes,
            "labels": pred_labels,
            "scores": pred_scores
        }
        
        # Apply NMS
        filtered_predictions = nms_processor(predictions)
        
        # Get image path
        image_path = os.path.join(image_dir, f"{image_id}.png")
        
        # Create visualization
        save_path = os.path.join(output_dir, f"{image_id}.png")
        visualize_predictions(image_path, filtered_predictions, save_path)
        
    print(f"Visualizations completed. Check {output_dir}")
 
# Example usage
if __name__ == "__main__":
    # Replace these with your actual paths
    
    visualize_from_csv()