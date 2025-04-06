import torch
import numpy as np
from torchvision.ops import nms

 
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