if __name__ == "__main__":
    from ultralytics import YOLO

    # Load model
    # model = YOLO("configs/yolo/yolov11.yaml", task="detect").load("checkpoints/yolo/yolo11x.pt")  # build from YAML and transfer weights
    model = YOLO("configs/yolo/yolov11.yaml", task="detect").load("runs/detect/train6/weights/best.pt")  # build from YAML and load weights
    # Train model
    model.train(data="configs/yolo/teeth_diagnosis.yaml", epochs=100, imgsz=640)
