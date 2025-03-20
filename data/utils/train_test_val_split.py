import json
import pandas as pd
import os
from sklearn.model_selection import train_test_split
 
# Convert COCO bbox format (x, y, w, h) → Pascal VOC format (x_min, y_min, x_max, y_max)
def xywh_to_xyxy(bbox):
    return [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
 
# Process dataset for VOC (Nests annotations inside images)
def process_voc_data(dataset, image_ids):
    images = []
    for image in dataset["images"]:
        if image["id"] in image_ids:
            # Create deep copies of annotations for this image
            annotations = []
            for original_ann in dataset["annotations"]:
                if original_ann["image_id"] == image["id"]:
                    # Create a copy of the annotation
                    ann = original_ann.copy()
                    # Create a copy of the bbox and convert it to VOC format
                    bbox_copy = ann["bbox"].copy() if isinstance(ann["bbox"], list) else ann["bbox"]
                    ann["bbox"] = xywh_to_xyxy(bbox_copy)
                    annotations.append(ann)
            
            # Create a copy of the image
            image_copy = image.copy()
            # Add annotations to the image copy
            image_copy["annotations"] = annotations
            images.append(image_copy)
    return images
 
# Process dataset for COCO (Keeps images and annotations separate)
def process_coco_data(dataset, image_ids):
    # Create deep copies of images and annotations to avoid modifying the original dataset
    images = [img.copy() for img in dataset["images"] if img["id"] in image_ids]
    
    annotations = []
    for original_ann in dataset["annotations"]:
        if original_ann["image_id"] in image_ids:
            # Create a deep copy of the annotation
            ann = original_ann.copy()
            # Create a deep copy of the bbox to avoid modifying the original
            if "bbox" in ann:
                ann["bbox"] = ann["bbox"].copy() if isinstance(ann["bbox"], list) else ann["bbox"]
            annotations.append(ann)
    
    # Create deep copies of category lists
    categories_1 = [cat.copy() for cat in dataset.get("categories_1", [])]
    categories_2 = [cat.copy() for cat in dataset.get("categories_2", [])]
    
    return images, annotations, categories_1, categories_2

# Compute statistics
def flatten_annotations(data):
    rows = []
    for img in data:
        for ann in img["annotations"]:
            rows.append({"image_id": img["id"], "bbox": ann["bbox"]})
    return pd.DataFrame(rows)
 
# Save dataset in COCO JSON format (Single File) without duplicate annotations
def save_coco_json(images, annotations, categories_1, categories_2, dataset, save_path):
    dataset_copy = dataset.copy()
    
    # Ensure images don't already contain annotations
    clean_images = []
    for img in images:
        clean_img = img.copy()
        if "annotations" in clean_img:
            del clean_img["annotations"]
        clean_images.append(clean_img)
    
    dataset_copy["images"] = clean_images
    dataset_copy["annotations"] = annotations
    dataset_copy["categories_1"] = categories_1
    dataset_copy["categories_2"] = categories_2
 
    with open(save_path, "w") as f:
        json.dump(dataset_copy, f, indent=4)
 
# Main function
def main():
    dataset_types = ["quadrant_enumeration", "quadrant_enumeration_disease"]
    base_path = "datasets/coco/"
 
    for dataset_type in dataset_types:
        json_path = os.path.join(base_path, f"{dataset_type}/train_{dataset_type}.json")
 
        with open(json_path) as f:
            dataset = json.load(f)
        
        # Split image IDs first to ensure VOC and COCO use the same splits
        all_image_ids = [img["id"] for img in dataset["images"]]
        train_ids, temp_ids = train_test_split(all_image_ids, test_size=0.25, random_state=42)
        val_ids, test_ids = train_test_split(temp_ids, test_size=0.4, random_state=42)
        
        # Convert to sets for faster lookups
        train_id_set = set(train_ids)
        val_id_set = set(val_ids)
        test_id_set = set(test_ids)
        
        # Process and save VOC format
        train_voc_data = process_voc_data(dataset, train_id_set)
        val_voc_data = process_voc_data(dataset, val_id_set)
        test_voc_data = process_voc_data(dataset, test_id_set)
        
        # Save VOC to CSV
        train_df = pd.DataFrame(train_voc_data)
        val_df = pd.DataFrame(val_voc_data)
        test_df = pd.DataFrame(test_voc_data)
        
        train_df.to_csv(os.path.join(base_path, f"{dataset_type}/{dataset_type}_voc_train.csv"), index=False)
        val_df.to_csv(os.path.join(base_path, f"{dataset_type}/{dataset_type}_voc_val.csv"), index=False)
        test_df.to_csv(os.path.join(base_path, f"{dataset_type}/{dataset_type}_voc_test.csv"), index=False)
        print(f"[VOC] Train/Val/Test CSV saved for {dataset_type}.")
        
        # Process and save COCO format
        train_images, train_annotations, categories_1, categories_2 = process_coco_data(dataset, train_id_set)
        val_images, val_annotations, _, _ = process_coco_data(dataset, val_id_set)
        test_images, test_annotations, _, _ = process_coco_data(dataset, test_id_set)
        
        # Save COCO JSON
        save_coco_json(train_images, train_annotations, categories_1, categories_2, dataset,
                      os.path.join(base_path, f"{dataset_type}/{dataset_type}_coco_train.json"))
        
        save_coco_json(val_images, val_annotations, categories_1, categories_2, dataset,
                      os.path.join(base_path, f"{dataset_type}/{dataset_type}_coco_val.json"))
        
        save_coco_json(test_images, test_annotations, categories_1, categories_2, dataset,
                      os.path.join(base_path, f"{dataset_type}/{dataset_type}_coco_test.json"))
        
        print(f"[COCO] Train/Val/Test JSON saved for {dataset_type}.")
        
        # Compute and print statistics
        train_ann_df = flatten_annotations(train_voc_data)
        val_ann_df = flatten_annotations(val_voc_data)
        test_ann_df = flatten_annotations(test_voc_data)
        
        print(f"Train: {train_ann_df['image_id'].nunique()} unique images, {train_ann_df.shape[0]} bboxes")
        print(f"Val: {val_ann_df['image_id'].nunique()} unique images, {val_ann_df.shape[0]} bboxes")
        print(f"Test: {test_ann_df['image_id'].nunique()} unique images, {test_ann_df.shape[0]} bboxes")
 
if __name__ == "__main__":
    main()