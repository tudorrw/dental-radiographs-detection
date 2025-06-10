import torch
import numpy as np


from utils.get_models import FineTunedModels
import models.dino.datasets.transforms as DT

device = "cuda" if torch.cuda.is_available() else "cpu"

models = FineTunedModels(device)
yolo_model = None

rcnn_model = models.get_faster_rcnn_model()
retinanet_model = models.get_retinanet_model()
dino_model, postprocessors = models.get_dino_model()
yolo_model = models.get_yolo_model()

def predict_faster_rcnn(image):
    image_tensor = torch.Tensor(np.array(image)).permute(2, 0, 1) / 255.0
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        prediction = rcnn_model([image_tensor])[0]

    boxes_cpu = prediction["boxes"].cpu().numpy()
    scores_cpu = prediction["scores"].cpu().numpy()
    labels_cpu = prediction["labels"].cpu().numpy()

    return {
        "boxes": boxes_cpu,
        "scores": scores_cpu,
        "labels": labels_cpu
    }

def predict_retinanet(image):
    image_tensor = torch.Tensor(np.array(image)).permute(2, 0, 1) / 255.0
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        prediction = retinanet_model([image_tensor])[0]

    boxes_cpu = prediction["boxes"].cpu().numpy()
    scores_cpu = prediction["scores"].cpu().numpy()
    labels_cpu = prediction["labels"].cpu().numpy()

    return {
        "boxes": boxes_cpu,
        "scores": scores_cpu,
        "labels": labels_cpu
    }
    
    
def predict_yolo(image):
    results = yolo_model.predict(image, conf=0.5)[0]

    boxes_cpu = results.boxes.xyxy.cpu().numpy()
    scores_cpu = results.boxes.conf.cpu().numpy()
    labels_cpu = results.boxes.cls.cpu().numpy()
    
    return {
        "boxes": boxes_cpu,
        "scores": scores_cpu,
        "labels": labels_cpu
    }
    
def predict_dino(image):
    image_np = np.array(image)
    image_shape = image_np.shape[:2]
    
    transforms = DT.Compose([
        DT.RandomResize([800], max_size=1333),
        DT.ToTensor(),
        DT.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image_tensor = image.copy()
    image_tensor, _ = transforms(image_tensor, None)
    if device == "cuda":
        image_tensor = image_tensor.cuda()
            
    outputs = dino_model(image_tensor.unsqueeze(0))
    scale = torch.tensor([image_shape])
    if device == "cuda":
        scale = scale.cuda()
            
    output = postprocessors["bbox"](outputs, scale)[0]

    # Post-process predictions
    scores = output["scores"].cpu().numpy()
    labels = (output["labels"] +1).cpu().numpy()
    boxes = output["boxes"].cpu().numpy()
    
    return {
        "boxes": boxes,
        "scores": scores,
        "labels": labels
    }
