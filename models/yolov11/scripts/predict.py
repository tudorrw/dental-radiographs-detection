import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import yaml
import csv
import glob
from ultralytics import YOLO
from tqdm import tqdm


BASE_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)))
CFG_PATH = os.path.join(BASE_FOLDER, "..", "teeth_enumeration.yaml")


if __name__ == "__main__":

    with open(CFG_PATH, "r") as f:
        config = yaml.safe_load(f)
    print(config)

    test_image_dir = os.path.join("dataset", "yolo", config["test"])
    image_paths = glob.glob(os.path.join(test_image_dir, "*.png"))

    output_dir = os.path.join("results", "yolo", "train")
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = os.path.join(output_dir, "predictions_results.csv")
    file_exists = os.path.isfile(csv_path)

    checkpoint = os.path.join("runs", "detect", "train", "weights", "best.pt")
    model = YOLO(checkpoint)

    with open(csv_path, 'a', newline='') as csvfile:
        fieldnames = ['image_id', 'prediction_boxes', 'prediction_scores', 'prediction_labels']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header if file doesn't exist
        if not file_exists:
            writer.writeheader()
    
        for image_path in tqdm(image_paths):
            image_id = os.path.basename(image_path).split(".")[0]
            results = model(image_path, conf=0.5)
            # results is a list; use the first result.
            result = results[0]
            # Extract boxes (xyxy), confidences, and class labels.
            boxes = result.boxes.xyxy.cpu().numpy().tolist()
            scores = result.boxes.conf.cpu().numpy().tolist()
            labels = list(map(int, result.boxes.cls.cpu().numpy().tolist()))
            writer.writerow({
                "image_id": image_id,
                "prediction_boxes": str(boxes),
                "prediction_scores": str(scores),
                "prediction_labels": str(labels),
            })






    

