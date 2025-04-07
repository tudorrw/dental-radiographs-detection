import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from ultralytics import YOLO

def train_yolo_model():
    BASE_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    MODEL_PATH = os.path.join(BASE_FOLDER, "..", "yolo11x.yaml")
    CFG_PATH = os.path.join(BASE_FOLDER, "..", "teeth_enumeration.yaml")
    SAVE_PATH = os.path.join(BASE_FOLDER, "..", "..", "..", "checkpoints", "yolo")

    model = YOLO(MODEL_PATH, task="detect").load("../../checkpoints/yolo_helpers/yolo11x.pt")  # build from YAML and transfer weights
    # model.add_callback("on_pretrain_routine_end", callback_custom_albumentations)    # Train model
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    model.train(data=CFG_PATH, epochs=50, imgsz=640, batch=2, device=0, verbose=True, perspective=0.001)  # train

if __name__ == "__main__":
    train_yolo_model()



