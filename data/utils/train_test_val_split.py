import json
import pandas as pd
import os
import shutil
import random
from sklearn.model_selection import train_test_split
from utils.mapper import ToothLabelMapper

# Convert COCO bbox format (x, y, w, h) → Pascal VOC format (x_min, y_min, x_max, y_max)
def xywh_to_xyxy(bbox):
    return [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
 
def xywh_to_xcycwh(image_width, image_height, bbox):
    x_center = (bbox[0] + bbox[2] / 2.0) / image_width
    y_center = (bbox[1] + bbox[3] / 2.0) / image_height
    w = bbox[2] / image_width
    h = bbox[3] / image_height
    return f"{x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"

# Helper functions for file operations
def load_json(file_path):
    with open(file_path) as f:
        return json.load(f)

def save_json(file_path, data):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

def mkdirs(dir_path):
    os.makedirs(dir_path, exist_ok=True)

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
def process_coco_data_quadrant_emun(dataset, image_ids):
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

def process_coco_data_quadrant(dataset, image_ids):
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
    categories = [cat.copy() for cat in dataset.get("categories", [])]
    
    return images, annotations, categories

# Process dataset for DINO
def process_dino_data(dataset, image_ids):
    # Create deep copies of images and annotations to avoid modifying the original dataset
    images = [img.copy() for img in dataset["images"] if img["id"] in image_ids]
    
    # Initialize tooth label mapper
    label_mapper = ToothLabelMapper()
    
    annotations = []
    for original_ann in dataset["annotations"]:
        if original_ann["image_id"] in image_ids:
            # Create a deep copy of the annotation
            ann = original_ann.copy()
            # Create a copy of the bbox
            if "bbox" in ann:
                ann["bbox"] = ann["bbox"].copy() if isinstance(ann["bbox"], list) else ann["bbox"]
            
            # Convert quadrant enumeration to single category ID for DINO using the same mapping as YOLO
            if "category_id_1" in ann and "category_id_2" in ann:
                category_id_1 = ann.pop("category_id_1")
                category_id_2 = ann.pop("category_id_2")
                # Calculate tooth ID (same as in YOLO process)
                tooth_id = category_id_1 * 10 + category_id_2 + 1
                # Use the tooth label mapper to get a consistent mapping
                # mapped_class_id = int(label_mapper.encode([tooth_id])[0])
                ann["category_id"] = category_id_1 * 8 + category_id_2
            
            annotations.append(ann)
    
    # Create categories for DINO using the tooth mapper
    categories = [{"id": i, "name": str(i + 1), "supercategory": str(i + 1)} for i in range(32)]
    
    return images, annotations, categories

# Process COCO to YOLO conversion
def process_yolo_data(dataset, image_ids, origin_path, dataset_type, output_dir):
    # Create YOLO directories for labels only (images will stay in origin)
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_dir, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, "labels"), exist_ok=True)

    label_mapper = ToothLabelMapper()  
    
    # Process for each split
    for split, split_ids in [("train", train_id_set), ("val", val_id_set), ("test", test_id_set)]:
        labels_dir = os.path.join(output_dir, split, "labels")
        images_dir = os.path.join(output_dir, split, "images")

        for img_id in split_ids:
            # Get image information
            image_info = next((img for img in dataset["images"] if img["id"] == img_id), None)
            if not image_info:
                continue
                
            image_width, image_height = image_info["width"], image_info["height"]
            image_filename = image_info["file_name"]
            
            src_image_path = os.path.join(origin_path, dataset_type, "xrays", image_filename)
            dst_image_path = os.path.join(images_dir, image_filename)
            if not os.path.exists(dst_image_path):
                shutil.copy2(src_image_path, dst_image_path)
            # Create YOLO label file
            label_filename = os.path.splitext(image_filename)[0] + ".txt"
            label_path = os.path.join(labels_dir, label_filename)
            
            with open(label_path, "w") as f:
                for ann in dataset["annotations"]:
                    if ann["image_id"] == img_id:
                        # Get category ID and convert to YOLO class ID
                        category_id_1 = ann["category_id_1"]
                        category_id_2 = ann["category_id_2"]
                        tooth_id = category_id_1 * 10 + category_id_2 + 1
                        mapped_class_id = int(label_mapper.encode([tooth_id])[0])

                        # Convert bbox to YOLO format
                        bbox = ann["bbox"]
                        yolo_bbox = xywh_to_xcycwh(image_width, image_height, bbox)
                        
                        # Write to file (class_id x_center y_center width height)
                        f.write(f"{mapped_class_id} {yolo_bbox}\n")  # Subtract 1 for 0-indexed classes in YOLO
                        
    print(f"[YOLO] Train/Val/Test labels saved for {dataset_type} in {output_dir}")

# Compute statistics
def flatten_annotations(data):
    rows = []
    for img in data:
        for ann in img["annotations"]:
            rows.append({"image_id": img["id"], "bbox": ann["bbox"]})
    return pd.DataFrame(rows)
 
# Save dataset in COCO JSON format (Single File) without duplicate annotations
def save_coco_json_quadrant_enum(images, annotations, categories_1, categories_2, dataset, save_path):
    dataset_copy = dataset.copy()

    dataset_copy["images"] = images
    dataset_copy["annotations"] = annotations
    dataset_copy["categories_1"] = categories_1
    dataset_copy["categories_2"] = categories_2
 
    with open(save_path, "w") as f:
        json.dump(dataset_copy, f, indent=4)

# Save dataset in COCO JSON format for DINO
def save_coco_json_dino(images, annotations, categories, save_path):
    dataset = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }
    
    with open(save_path, "w") as f:
        json.dump(dataset, f, indent=4)

# Process DINO format with directory structure for COCO
def process_dino_data_with_dirs(dataset, image_ids, origin_path, dataset_type, output_dir):
    # Create required directories
    train_dir = os.path.join(output_dir, "train2017")
    val_dir = os.path.join(output_dir, "val2017")
    test_dir = os.path.join(output_dir, "test2017")
    annotations_dir = os.path.join(output_dir, "annotations")
    
    for directory in [train_dir, val_dir, test_dir, annotations_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Process for train, val, test splits
    train_images, train_annotations, train_categories = process_dino_data(dataset, train_id_set)
    val_images, val_annotations, val_categories = process_dino_data(dataset, val_id_set)
    test_images, test_annotations, test_categories = process_dino_data(dataset, test_id_set)
    
    # Copy images to respective directories
    for split, split_images, split_dir in [
        ("train", train_images, train_dir),
        ("val", val_images, val_dir),
        ("test", test_images, test_dir)
    ]:
        for img in split_images:
            src_path = os.path.join(origin_path, dataset_type, "xrays", img["file_name"])
            dst_path = os.path.join(split_dir, img["file_name"])
            if not os.path.exists(dst_path):
                shutil.copy2(src_path, dst_path)
    
    # Save annotations
    save_coco_json_dino(
        train_images, train_annotations, train_categories,
        os.path.join(annotations_dir, "instances_train2017.json")
    )
    
    save_coco_json_dino(
        val_images, val_annotations, val_categories,
        os.path.join(annotations_dir, "instances_val2017.json")
    )
    
    save_coco_json_dino(
        test_images, test_annotations, test_categories,
        os.path.join(annotations_dir, "instances_test2017.json")
    )
    
    print(f"[DINO] Train/Val/Test JSON and images saved for {dataset_type} in {output_dir}")

# Main function
def main():
    
    dataset_type = "quadrant_enumeration"
    global train_id_set, val_id_set, test_id_set
    origin_path = "dataset/origin/"
    coco_base_path_detr = "dataset/coco/detr"
    coco_base_path_dino = "dataset/coco/dino"
    pascal_voc_base_path = "dataset/pascal_voc/"
    yolo_base_path = "dataset/yolo/classic"

    for path in [coco_base_path_detr, coco_base_path_dino, pascal_voc_base_path, yolo_base_path]:
        os.makedirs(path, exist_ok=True)
 
    os.makedirs(os.path.join(coco_base_path_detr, dataset_type), exist_ok=True)
    os.makedirs(os.path.join(coco_base_path_dino, dataset_type), exist_ok=True)
    os.makedirs(os.path.join(pascal_voc_base_path, dataset_type), exist_ok=True)

    json_path = os.path.join(origin_path, f"{dataset_type}/train_{dataset_type}.json")

    with open(json_path) as f:
        dataset = json.load(f)

    # Split image IDs first to ensure VOC and COCO use the same splits
    all_image_ids = [img["id"] for img in dataset["images"]]
    train_ids, temp_ids = train_test_split(all_image_ids, test_size=0.20, random_state=42)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)

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
    
    train_df.to_csv(os.path.join(pascal_voc_base_path, f"{dataset_type}/{dataset_type}_voc_train.csv"), index=False)
    val_df.to_csv(os.path.join(pascal_voc_base_path, f"{dataset_type}/{dataset_type}_voc_val.csv"), index=False)
    test_df.to_csv(os.path.join(pascal_voc_base_path, f"{dataset_type}/{dataset_type}_voc_test.csv"), index=False)
    print(f"[VOC] Train/Val/Test CSV saved for {dataset_type}.")
    
    # Process and save DETR format
    train_images, train_annotations, categories_1, categories_2 = process_coco_data_quadrant_emun(dataset, train_id_set)
    val_images, val_annotations, _, _ = process_coco_data_quadrant_emun(dataset, val_id_set)
    test_images, test_annotations, _, _ = process_coco_data_quadrant_emun(dataset, test_id_set)
    
    # Save COCO JSON for DETR
    save_coco_json_quadrant_enum(train_images, train_annotations, categories_1, categories_2, dataset,
                os.path.join(coco_base_path_detr, f"{dataset_type}/{dataset_type}_coco_train.json"))
    
    save_coco_json_quadrant_enum(val_images, val_annotations, categories_1, categories_2, dataset,
                os.path.join(coco_base_path_detr, f"{dataset_type}/{dataset_type}_coco_val.json"))
    
    save_coco_json_quadrant_enum(test_images, test_annotations, categories_1, categories_2, dataset,
                os.path.join(coco_base_path_detr, f"{dataset_type}/{dataset_type}_coco_test.json"))

    print(f"[COCO DETR] Train/Val/Test JSON saved for {dataset_type}.")

    # Process YOLO format
    process_yolo_data(dataset, all_image_ids, origin_path, dataset_type, yolo_base_path)
    
    # Process DINO format
    process_dino_data_with_dirs(dataset, all_image_ids, origin_path, dataset_type, 
                                os.path.join(coco_base_path_dino, dataset_type))
    
    # Compute and print statistics
    train_ann_df = flatten_annotations(train_voc_data)
    val_ann_df = flatten_annotations(val_voc_data)
    test_ann_df = flatten_annotations(test_voc_data)
    
    print(f"Train: {train_ann_df['image_id'].nunique()} unique images, {train_ann_df.shape[0]} bboxes")
    print(f"Val: {val_ann_df['image_id'].nunique()} unique images, {val_ann_df.shape[0]} bboxes")
    print(f"Test: {test_ann_df['image_id'].nunique()} unique images, {test_ann_df.shape[0]} bboxes")

if __name__ == "__main__":
    main()