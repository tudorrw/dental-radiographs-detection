import json
import pandas as pd
import os
import argparse
from sklearn.model_selection import train_test_split

# Convert COCO bbox format (x, y, w, h) → Pascal VOC format (x_min, y_min, x_max, y_max)
def xywh_to_xyxy(bbox):
    return [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]

# Process dataset: Nest annotations inside images
def nest_ann_into_images(dataset):
    images = []
    for image in dataset["images"]:
        annotations = [ann for ann in dataset["annotations"] if ann["image_id"] == image["id"]]
        for ann in annotations:
            ann["bbox"] = xywh_to_xyxy(ann["bbox"])  # Convert bbox format
        image["annotations"] = annotations
        images.append(image)
    return images

# Split dataset into train/val/test
def get_train_val_test_split(data, train_ratio=0.75, val_ratio=0.15, test_ratio=0.10):
    train_data, temp_data = train_test_split(data, test_size=(1 - train_ratio), random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=test_ratio / (test_ratio + val_ratio), random_state=42)
    return train_data, val_data, test_data

# Keep flatten_annotations as provided
def flatten_annotations(data):
    rows = []
    for img in data:
        for ann in img["annotations"]:
            rows.append({"image_id": img["id"], "bbox": ann["bbox"]})
    return pd.DataFrame(rows)

# Save dataset in VOC CSV format
def save_voc_csv(data, save_path):
    df = flatten_annotations(data)
    df.to_csv(save_path, index=False)

# Save dataset in COCO JSON format
def save_coco_json(data, dataset, save_path):
    dataset_copy = dataset.copy()
    dataset_copy["images"] = data
    dataset_copy["annotations"] = [ann for img in data for ann in img["annotations"]]
    
    with open(save_path, "w") as f:
        json.dump(dataset_copy, f, indent=4)

# Main function
def main(format_type):
    dataset_types = ["quadrant_enumeration", "quadrant_enumeration_disease"]
    base_path = "datasets/coco/"

    for dataset_type in dataset_types:
        json_path = os.path.join(base_path, f"{dataset_type}/train_{dataset_type}.json")

        with open(json_path) as f:
            dataset = json.load(f)

        processed_data = nest_ann_into_images(dataset)
        train_data, val_data, test_data = get_train_val_test_split(processed_data)

        # Save in selected format
        if format_type == "voc":

            train_df = pd.DataFrame(train_data)
            val_df = pd.DataFrame(val_data)
            test_df = pd.DataFrame(test_data)

            # Save to CSV
            train_df.to_csv(os.path.join(base_path, f"{dataset_type}/{dataset_type}_voc_train.csv"), index=False)
            val_df.to_csv(os.path.join(base_path, f"{dataset_type}/{dataset_type}_voc_val.csv"), index=False)
            test_df.to_csv(os.path.join(base_path, f"{dataset_type}/{dataset_type}__voc_test.csv"), index=False)
            print(f"[VOC] Train/Val/Test split saved for {dataset_type}.")

                    # Compute statistics
            train_ann_df = flatten_annotations(train_data)
            val_ann_df = flatten_annotations(val_data)
            test_ann_df = flatten_annotations(test_data)

            print(f"Train: {train_ann_df['image_id'].nunique()} unique images, {train_ann_df.shape[0]} bboxes")
            print(f"Val: {val_ann_df['image_id'].nunique()} unique images, {val_ann_df.shape[0]} bboxes")
            print(f"Test: {test_ann_df['image_id'].nunique()} unique images, {test_ann_df.shape[0]} bboxes")
        
        elif format_type == "coco":
            save_coco_json(train_data, dataset, os.path.join(base_path, f"{dataset_type}/{dataset_type}_coco_train.json"))
            save_coco_json(val_data, dataset, os.path.join(base_path, f"{dataset_type}/{dataset_type}_coco_val.json"))
            save_coco_json(test_data, dataset, os.path.join(base_path, f"{dataset_type}/{dataset_type}_coco_test.json"))
            print(f"[COCO] Train/Val/Test split saved for {dataset_type}.")
        
        else:
            print(f"Invalid format type: {format_type}. Please use 'voc' or 'coco'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert dataset into VOC or COCO format")
    parser.add_argument("format", type=str, choices=["voc", "coco"], help="Format type: voc or coco")
    args = parser.parse_args()

    main(args.format)
