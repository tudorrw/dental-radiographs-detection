import json
import os
import shutil
import numpy as np
from tqdm import tqdm
import random
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


class StratifiedCocoToYolo:
    def __init__(self, coco_json_path, coco_image_folder, yolo_output_dir, val_size=0.2):
        self.coco_json_path = coco_json_path
        self.coco_image_folder = coco_image_folder
        self.yolo_output_dir = yolo_output_dir
        self.val_size = val_size

        # Load COCO data
        self.coco_data = self.load_json(self.coco_json_path)
        self.image_data = {img["id"]: img for img in self.coco_data["images"]}
        self.annotations = self.coco_data["annotations"]

        self.create_output_folders()

    def load_json(self, file_path):
        with open(file_path) as f:
            return json.load(f)

    def create_output_folders(self):
        for split in ["train", "val"]:
            os.makedirs(os.path.join(self.yolo_output_dir, split, "images"), exist_ok=True)
            os.makedirs(os.path.join(self.yolo_output_dir, split, "labels"), exist_ok=True)

    def coco_to_yolo_bbox(self, image_width, image_height, bbox):
        x_center = (bbox[0] + bbox[2] / 2.0) / image_width
        y_center = (bbox[1] + bbox[3] / 2.0) / image_height
        w = bbox[2] / image_width
        h = bbox[3] / image_height
        return f"{x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"


    def perform_stratified_split(self):
        """
        Perform stratified split while preserving the proportion of each class.
        Also handles images with no bounding boxes (background images).
        """

        # Group annotations by image and collect all labels per image
        image_labels = {}  # Stores all class labels per image
        background_images = set(self.image_data.keys())  # Assume all images are background initially

        for ann in tqdm(self.annotations, desc="Processing Annotations"):
            img_id = ann["image_id"]
            cat_id = ann["category_id_3"]
            if img_id not in image_labels:
                image_labels[img_id] = set()
            image_labels[img_id].add(cat_id)  # Multiple labels per image

            if img_id in background_images:
                background_images.remove(img_id)  # This image contains objects, so it's not background

        # Convert labels to binary format (multi-label classification)
        all_classes = sorted(set([ann["category_id_3"] for ann in self.annotations]))  # Unique classes
        num_classes = len(all_classes)
        image_ids = np.array(list(image_labels.keys()))

        # Create binary matrix: each row corresponds to an image, each column to a class
        labels_matrix = np.zeros((len(image_ids), num_classes), dtype=int)
        for i, img_id in enumerate(image_ids):
            for cat_id in image_labels[img_id]:
                labels_matrix[i, cat_id] = 1  # Mark class presence

        # Perform stratified multi-label split for images with objects
        msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=self.val_size, random_state=42)
        train_idx, val_idx = next(msss.split(image_ids, labels_matrix))

        train_ids, val_ids = set(image_ids[train_idx]), set(image_ids[val_idx])

        # Evenly distribute background images
        background_images = list(background_images)  # Convert to list
        random.shuffle(background_images)
        split_idx = int(len(background_images) * (1 - self.val_size))

        train_bg = set(background_images[:split_idx])
        val_bg = set(background_images[split_idx:])

        # Merge object and background images
        train_ids.update(train_bg)
        val_ids.update(val_bg)

        print(f"Total images: {len(self.image_data)}, Train: {len(train_ids)}, Val: {len(val_ids)}")
        print(f"Background Images - Train: {len(train_bg)}, Val: {len(val_bg)}")

        # print(train_ids, "valinaaffsevewewvb", val_ids)
        self.create_yolo_files(train_ids, val_ids)
        




    def create_yolo_files(self, train_ids, val_ids):
        for img_id in tqdm(train_ids, desc="Processing Train Data"):

            self.process_image(img_id, "train")

        for img_id in tqdm(val_ids, desc="Processing Validation Data"):
            self.process_image(img_id, "val")


    def process_image(self, img_id, split):
        image_info = self.image_data[img_id]
        image_filename = image_info["file_name"]
        image_width, image_height = image_info["width"], image_info["height"]

        # Copy image to the split folder
        original_image_path = os.path.join(self.coco_image_folder, image_filename)
        yolo_image_path = os.path.join(self.yolo_output_dir, split, "images", image_filename)
        shutil.copy(original_image_path, yolo_image_path)

        # Create label file
        label_filename = image_info["file_name"][:-4] + ".txt"
        yolo_label_path = os.path.join(self.yolo_output_dir, split, "labels", label_filename)

        with open(yolo_label_path, "w") as label_file:
            for annotation in self.annotations:
                if annotation["image_id"] == img_id:
                    class_id = annotation["category_id_3"]
                    bbox = annotation["bbox"]
                    yolo_bbox = self.coco_to_yolo_bbox(image_width, image_height, bbox)
                    label_file.write(f"{class_id} {yolo_bbox}\n")

    def display_class_distribution(self):
        train_labels, val_labels = [], []

        for img_id in self.image_data:
            for annotation in self.annotations:
                if annotation["image_id"] == img_id:
                    if os.path.exists(os.path.join(self.yolo_output_dir, "train", "images", self.image_data[img_id]["file_name"])):
                        train_labels.append(annotation["category_id_3"])
                    else:
                        val_labels.append(annotation["category_id_3"])

        print("\nClass Distribution:")
        print(f"Train: {dict(zip(*np.unique(train_labels, return_counts=True)))}")
        print(f"Val: {dict(zip(*np.unique(val_labels, return_counts=True)))}")

if __name__ == "__main__":
    coco_json_path = r"../dataset/origin/quadrant_enumeration_disease/train_quadrant_enumeration_disease.json"
    coco_image_folder = r"../dataset/origin/quadrant_enumeration_disease/xrays"
    yolo_output_dir = r"../dataset/yolo"

    converter = StratifiedCocoToYolo(coco_json_path, coco_image_folder, yolo_output_dir, val_size=0.2)
    converter.perform_stratified_split()
    converter.display_class_distribution()

