
import json
import random
import os
from tqdm import tqdm
import shutil

class CocoYoloFormatConvesion:

    def __init__(self, coco_json_path, coco_image_folder, yolo_output_dir):
        self.coco_json_path = coco_json_path
        self.coco_image_folder = coco_image_folder
        self.yolo_output_dir = yolo_output_dir

        for split in ["train", "val"]:
            os.makedirs(os.path.join(yolo_output_dir, split, "labels"), exist_ok=True)
            os.makedirs(os.path.join(yolo_output_dir, split, "images"), exist_ok=True)

        self.coco_data = self.read_raw_json(self.coco_json_path)
        self.image_data = {img["id"]: img for img in self.coco_data["images"]}
        self.annotations = self.coco_data["annotations"]
        
        # Perform train val split function
        self.train_val_split()

    def read_raw_json(self, file_path):
        with open(file_path) as json_data:
            data = json.load(json_data)
        return data
    
    def coco_to_yolo_bbox(self, image_width, image_height, bbox):
        x_min, y_min, bbox_width, bbox_height = bbox

        x_center =(x_min + bbox_width / 2.0) / image_width
        y_center = (y_min + bbox_height / 2.0) / image_height
        w = bbox_width / image_width
        h = bbox_height / image_height
        
        return f"{x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"
    
    
    def train_val_split(self):

        #Shuffle the image ids
        image_ids = list(self.image_data.keys())
        random.shuffle(image_ids)

        #Split the data into train and validation
        split_index = int(len(image_ids) * 0.8)
        self.train_ids = set(image_ids[:split_index])
        self.val_ids = set(image_ids[split_index:])

    def annotation_to_yolo_label(self):
        
        for annotation in tqdm(self.annotations):

            image_id = annotation["image_id"]
            category_id = annotation["category_id_3"]  # Adjust according to your dataset
            bbox = annotation["bbox"]

            image_info = self.image_data[image_id]
            image_width, image_height = image_info["width"], image_info["height"]

            # Convert bbox format
            yolo_bbox = self.coco_to_yolo_bbox(image_width, image_height, bbox)

            # # Determine train/val split
            split = "train" if image_id in self.train_ids else "val"
            

            #remove the file extension
            image_name = image_info["file_name"][:-4]
            # # YOLO annotation file path
            yolo_label_path = os.path.join(yolo_output_dir, split, "labels", f"{image_name}.txt")
            

            # Write annotation in YOLO format
            with open(yolo_label_path, "a") as yolo_file:
                yolo_file.write(f"{category_id} {yolo_bbox}\n")

            # Copy images to the train/val folder
            original_image_path = os.path.join(coco_image_folder, image_info["file_name"])
            yolo_image_path = os.path.join(yolo_output_dir, split, "images", image_info["file_name"])
            
            if os.path.exists(original_image_path):
                shutil.copy(original_image_path, yolo_image_path)

    def convert(self):
        print("Starting the conversion process")
        self.annotation_to_yolo_label()
        print("Conversion process completed")




if __name__ == "__main__":

    coco_json_path = r"..\dataset\origin\quadrant_enumeration_disease\train_quadrant_enumeration_disease.json"
    coco_image_folder = r"..\dataset\origin\quadrant_enumeration_disease\xrays"
    yolo_output_dir = r"..\dataset\yolo"

    converter = CocoYoloFormatConvesion(coco_json_path, coco_image_folder, yolo_output_dir)
    converter.convert()