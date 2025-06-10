import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import uvicorn
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from utils.wbf import weighted_boxes_fusion
from utils.nms import UniqueClassNMSProcessor, CombinedNMS, ClassAgnosticNMS # If you're using class-wise NMS
from api.utils import read_convert_image, postprocess
from api.predictions import predict_faster_rcnn, predict_retinanet, predict_dino, predict_yolo

#for decoding the FDI numbers

app = FastAPI(title="Dental X-Ray Analysis API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


device = "cuda" if torch.cuda.is_available() else "cpu"



unique_class_nms = UniqueClassNMSProcessor(iou_threshold=0.55) 
class_agnostic_nms = ClassAgnosticNMS(iou_threshold=0.55, score_threshold=0.5)
combined_nms = CombinedNMS(iou_threshold=0.55, score_threshold=0.5)
combined_nms_2 = CombinedNMS(iou_threshold=0.55, score_threshold=0.3)

@app.post("/detections/faster-rcnn")
async def detect_teeth_faster_rcnn(file: UploadFile = File(...)):
    """
    Example endpoint that:
      - Reads the incoming X-ray image
      - Runs FasterRCNN
      - Applies a score filter + optional Unique-Class NMS
      - Decodes class indices to FDI numbers using label_mapper.decode
      - Returns the final bounding boxes + scores + FDI labels + PNG image
    """
    image = await read_convert_image(file)
    cpu_dict = predict_faster_rcnn(image)
    filtered = unique_class_nms(cpu_dict)
    
    return postprocess(image, filtered)



@app.post("/detections/yolo")
async def detect_teeth_yolov11(file: UploadFile = File(...)):
    image = await read_convert_image(file)
    cpu_dict = predict_yolo(image)
    filtered = combined_nms(cpu_dict)

    return postprocess(image, filtered)




@app.post("/detections/retinanet")
async def detect_teeth_retinanet(file: UploadFile = File(...)):
    """
    Example endpoint that:
      - Reads the incoming X-ray image
      - Runs RetinaNet
      - Applies a score filter + optional Unique-Class NMS
      - Decodes class indices to FDI numbers using label_mapper.decode
      - Returns the final bounding boxes + scores + FDI labels + PNG image
    """
    image = await read_convert_image(file)

    cpu_dict = predict_retinanet(image)
    filtered = combined_nms(cpu_dict)

    return postprocess(image, filtered)





@app.post("/detections/dino")
async def detect_teeth_dino(file: UploadFile = File(...)):
    image = await read_convert_image(file)
    cpu_dict = predict_dino(image)
    filtered = combined_nms_2(cpu_dict)


    return postprocess(image, filtered)


@app.post("/detections/meta-model")
async def detect_teeth_meta_model(file: UploadFile = File(...)):
    image = await read_convert_image(file)
    image_np = np.array(image)
    h, w = image_np.shape[:2]  # Get image height and width

    # Get predictions from all models
    rcnn = predict_faster_rcnn(image)
    retinanet = predict_retinanet(image)
    dino = predict_dino(image)
    
    # Normalize boxes by image dimensions
    def normalize_boxes(boxes):
        return np.array([
            [box[0]/w, box[1]/h, box[2]/w, box[3]/h] for box in boxes
        ])
    
    # Normalize boxes from each model
    rcnn_boxes = normalize_boxes(rcnn["boxes"]) if len(rcnn["boxes"]) > 0 else np.empty((0,4))
    retinanet_boxes = normalize_boxes(retinanet["boxes"]) if len(retinanet["boxes"]) > 0 else np.empty((0,4))
    dino_boxes = normalize_boxes(dino["boxes"]) if len(dino["boxes"]) > 0 else np.empty((0,4))
    
    # Prepare lists for WBF
    boxes_list = [rcnn_boxes, retinanet_boxes, dino_boxes]
    scores_list = [rcnn["scores"], retinanet["scores"], dino["scores"]]
    labels_list = [rcnn["labels"], retinanet["labels"], dino["labels"]]
    
    # Weights for each model (can be adjusted based on model performance)
    weights = [2, 1.65, 1.05]
    
    # Apply WBF
    boxes, scores, labels = weighted_boxes_fusion(
        boxes_list,
        scores_list,
        labels_list,
        weights=weights,
        iou_thr=0.55,
        conf_type="avg",
        skip_box_thr=0.5
    )
    
    # Filter boxes with scores below 0.55
    mask = scores >= 0.4
    boxes = boxes[mask]
    scores = scores[mask]
    labels = labels[mask]
    
    # Denormalize boxes back to original image coordinates
    denormalized_boxes = np.array([
        [box[0]*w, box[1]*h, box[2]*w, box[3]*h] for box in boxes
    ])
    
    combined_preds = {
        "boxes": denormalized_boxes,
        "scores": scores,
        "labels": labels
    }
    
    filtered = combined_nms(combined_preds)
    return postprocess(image, filtered)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
