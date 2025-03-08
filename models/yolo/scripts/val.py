from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("../../runs/detect/train/weights/best.pt", task="val")  # Load model
    metrics = model.val(data="../../configs/yolo/teeth_diagnosis.yaml", imgsz=640, batch=4, iou=0.6, conf=0.25)  # Validate model
    metrics.box.map
    metrics.box.map50
    metrics.box.map75
    metrics.box.maps
