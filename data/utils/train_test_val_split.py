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
            annotations = [ann for ann in dataset["annotations"] if ann["image_id"] == image["id"]]
            for ann in annotations:
                ann["bbox"] = xywh_to_xyxy(ann["bbox"])  # Convert bbox format for VOC
            image["annotations"] = annotations  # Nest annotations inside images
            images.append(image)
    return images
 
# Process dataset for COCO (Keeps images and annotations separate)
def process_coco_data(dataset, image_ids):
    images = [img for img in dataset["images"] if img["id"] in image_ids]
    annotations = [ann for ann in dataset["annotations"] if ann["image_id"] in image_ids]
    categories_1 = dataset.get("categories_1", [])
    categories_2 = dataset.get("categories_2", [])
    return images, annotations, categories_1, categories_2
 
# Split dataset into train/val/test
def get_train_val_test_split(data, train_ratio=0.75, val_ratio=0.15, test_ratio=0.10):
    train_data, temp_data = train_test_split(data, test_size=(1 - train_ratio), random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=test_ratio / (test_ratio + val_ratio), random_state=42)
    return train_data, val_data, test_data
 
# Compute statistics
def flatten_annotations(data):
    rows = []
    for img in data:
        for ann in img["annotations"]:
            rows.append({"image_id": img["id"], "bbox": ann["bbox"]})
    return pd.DataFrame(rows)
 
# Save dataset in COCO JSON format (Single File)
def save_coco_json(images, annotations, categories_1, categories_2, dataset, save_path):
    dataset_copy = dataset.copy()
    dataset_copy["images"] = images
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
        train_ids, temp_ids = train_test_split(all_image_ids, test_size=0.25, random_state=42) #75% train
        val_ids, test_ids = train_test_split(temp_ids, test_size=0.4, random_state=42) # 0.6 * 25 = 15% val, 10% test
        
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