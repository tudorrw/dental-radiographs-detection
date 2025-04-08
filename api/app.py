import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import io
import cv2
import base64
import torch
import uvicorn
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageDraw, ImageFont

from api.model import FineTunedModels
from utils.nms import UniqueClassNMSProcessor  # If you're using class-wise NMS
from api.utils import clahe, decode_teeth_numbers, draw_boxes, read_convert_image, output_json

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

# Example if you're applying class-wise NMS
device = "cuda" if torch.cuda.is_available() else "cpu"

models = FineTunedModels(device)

rcnn_model = models.get_faster_rcnn_model()

nms_processor = UniqueClassNMSProcessor(iou_threshold=0.5) 

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

    image_tensor = torch.Tensor(np.array(image)).permute(2, 0, 1) / 255.0
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        prediction = rcnn_model([image_tensor])[0]

    boxes_cpu = prediction["boxes"].cpu().numpy()
    scores_cpu = prediction["scores"].cpu().numpy()
    labels_cpu = prediction["labels"].cpu().numpy()

    cpu_dict = {
        "boxes": boxes_cpu,
        "scores": scores_cpu,
        "labels": labels_cpu
    }
    filtered = nms_processor(cpu_dict)

    final_boxes = filtered["boxes"]
    final_scores = filtered["scores"]
    final_labels = filtered["labels"]


    #decode the numeric indices to get FDI numbering
    predicted_quadrants, predicted_teeth, decoded_fdi_predicted_labels = decode_teeth_numbers(final_labels)

    #draw bounding boxes on a copy of the original image
    clahe_pil = clahe(image)
    img_str = draw_boxes(clahe_pil, final_boxes, predicted_quadrants, predicted_teeth)

    # 9) Return the final results
    return output_json(img_str, final_boxes, final_scores, decoded_fdi_predicted_labels)


yolo_model = models.get_yolo_model()

@app.post("/detections/yolov11")
async def detect_teeth_yolov11(file: UploadFile = File(...)):
    image = await read_convert_image(file)
    results = yolo_model.predict(image, conf=0.5)[0]

    boxes_cpu = results.boxes.xyxy.cpu().numpy()
    scores_cpu = results.boxes.conf.cpu().numpy()
    labels_cpu = results.boxes.cls.cpu().numpy()

    predicted_quadrants, predicted_teeth, decoded_fdi_predicted_labels = decode_teeth_numbers(labels_cpu)
    clahe_pil = clahe(image)
    img_str = draw_boxes(clahe_pil, boxes_cpu, predicted_quadrants, predicted_teeth)
    return output_json(img_str, boxes_cpu, scores_cpu, decoded_fdi_predicted_labels)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
