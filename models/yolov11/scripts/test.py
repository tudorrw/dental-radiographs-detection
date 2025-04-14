import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from ultralytics import YOLO

def test_yolo_model():
    BASE_FOLDER = os.path.dirname(os.path.abspath(__file__))
    # Path to your trained weights (the best.pt or last.pt)
    CHECKPOINTS_FOLDER = os.path.join(BASE_FOLDER, "..", "..", "..", "runs", "detect", "train", "weights", "best.pt")

    # Path to your data.yaml, which includes 'test:' 
    CFG_PATH = os.path.join(BASE_FOLDER, "..", "teeth_enumeration.yaml")

    # Load model with your trained weights
    model = YOLO(CHECKPOINTS_FOLDER)

    # Evaluate on the test set by specifying split='test'
    metrics = model.val(
        data=CFG_PATH,
        split="test",    # <-- This instructs it to use the test split
        imgsz=640,
        batch=4,
        iou=0.6,
        conf=0.25
    )

    # You can extract any of these values from 'metrics':
    print("mAP (50-95):", metrics.box.map)      # Overall mAP (mean of mAP50..mAP95)
    print("mAP50:", metrics.box.map50)
    print("mAP75:", metrics.box.map75)
    print("Per-class mAPs:", metrics.box.maps)  # list of mAP per-class

if __name__ == "__main__":
    test_yolo_model()
