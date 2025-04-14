import torch
import numpy as np
from torchvision.ops import nms, box_iou


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
        
        if len(boxes) == 0:
            return {"boxes": np.array([]), "labels": np.array([]), "scores": np.array([])}
        

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
    

class ClassAgnosticNMS:
    """
    Global class-agnostic NMS processor that ensures boxes with significant overlap are
    filtered regardless of their class labels, keeping only the highest scoring box.
    This is particularly helpful for tooth detection models like RetinaNet where
    overlapping boxes with different class labels can be problematic.
    """
    def __init__(self, iou_threshold=0.5, score_threshold=0.3):
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
    
    def __call__(self, output):
        """
        Process predictions with global class-agnostic NMS
        
        Args:
            output: dictionary with "boxes", "labels", "scores" keys
            
        Returns:
            output_processed: filtered predictions
        """
        boxes = torch.tensor(output["boxes"])
        scores = torch.tensor(output["scores"])
        labels = torch.tensor(output["labels"])
    
        # Filter by score threshold
        keep_indices = torch.where(scores > self.score_threshold)[0]
        if len(keep_indices) == 0:
            return {
                "boxes": np.array([]),
                "labels": np.array([]),
                "scores": np.array([])
            }
        
        boxes = boxes[keep_indices]
        scores = scores[keep_indices]
        labels = labels[keep_indices]
        
        # Sort by score (high to low)
        score_order = torch.argsort(scores, descending=True)
        boxes = boxes[score_order]
        scores = scores[score_order]
        labels = labels[score_order]
        
        # Initialize keep mask
        keep = torch.ones(len(boxes), dtype=torch.bool)
        
        # Apply class-agnostic NMS
        for i in range(len(boxes)):
            if not keep[i]:
                continue
                
            # Get IoU of this box with all remaining boxes
            current_box = boxes[i].unsqueeze(0)
            ious = box_iou(current_box, boxes[i+1:])
            
            # Identify boxes with IoU > threshold
            overlapping = torch.where(ious[0] > self.iou_threshold)[0]
            
            # Suppress overlapping boxes
            if len(overlapping) > 0:
                keep[i+1+overlapping] = False
        
        # Keep only the selected boxes
        final_indices = torch.where(keep)[0]
        boxes_processed = boxes[final_indices]
        labels_processed = labels[final_indices]
        scores_processed = scores[final_indices]
        
        # Convert to numpy for consistency
        output_processed = {
            "boxes": boxes_processed.cpu().numpy() if isinstance(boxes_processed, torch.Tensor) else boxes_processed,
            "labels": labels_processed.cpu().numpy() if isinstance(labels_processed, torch.Tensor) else labels_processed,
            "scores": scores_processed.cpu().numpy() if isinstance(scores_processed, torch.Tensor) else scores_processed
        }
        
        return output_processed
    


class CombinedNMS:
    def __init__(self, iou_threshold=0.5, score_threshold=0.3):
        self.agnostic_nms = ClassAgnosticNMS(iou_threshold=iou_threshold, score_threshold=score_threshold)
        self.unique_nms = UniqueClassNMSProcessor(iou_threshold=iou_threshold)

    def __call__(self, output):
        # first, it sorts all detection boxes by their confidence score in descending order. 
        filtered_output = self.agnostic_nms(output)
        # the detection box with the maximum confidence score is selected, 
        # all other detection boxes with significant overlap to that box are filtered out
        filtered_final = self.unique_nms(filtered_output)
        return filtered_final
    