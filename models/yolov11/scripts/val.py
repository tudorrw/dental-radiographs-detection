import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from ultralytics import YOLO

def val_yolo_model():
    BASE_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    CHECKPOINTS_FOLDER = os.path.join(BASE_FOLDER, "..", "..", "..", "runs", "detect", "train", "weights", "best.pt")
    CFG_PATH = os.path.join(BASE_FOLDER, "..", "teeth_enumeration.yaml")

    model = YOLO(CHECKPOINTS_FOLDER, task="val")  # build from YAML and transfer weights
    # model.add_callback("on_pretrain_routine_end", callback_custom_albumentations)    # Train model

    metrics = model.val(data=CFG_PATH, epochs=50, imgsz=640, batch=4, iou=0.6, conf=0.25)  # validate
    metrics.box.map
    metrics.box.map50
    metrics.box.map75
    metrics.box.maps

if __name__ == "__main__":
    val_yolo_model()
