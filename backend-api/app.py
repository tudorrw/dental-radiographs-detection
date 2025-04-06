import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import io
import cv2
import base64
import torch
import yaml
import uvicorn
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageDraw, ImageFont

from models.faster_rcnn.faster_rcnn import FasterRCNN
from utils.mapper import ToothLabelMapper
from utils.nms import UniqueClassNMSProcessor  # If you're using class-wise NMS

app = FastAPI(title="Dental X-Ray Analysis API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

label_mapper = ToothLabelMapper()
nms_processor = UniqueClassNMSProcessor(iou_threshold=0.5)  # Example if you're applying class-wise NMS
device = "cuda" if torch.cuda.is_available() else "cpu"

def get_model():
    # 1) Load config
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models",
        "faster_rcnn",
        "cfg.yaml"
    )
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 2) Load model checkpoint
    checkpoint_dict = dict(version="version_0", filename="epoch=44-val_loss=0.94.ckpt")
    checkpoint = os.path.join(
        config["checkpoints_path"],
        "faster_rcnn",
        checkpoint_dict["version"],
        checkpoint_dict["filename"]
    )
    print("Loading model from checkpoint:", checkpoint)

    model = FasterRCNN.load_from_checkpoint(checkpoint)
    model.eval()
    model.to(device)
    return model

model = get_model()

def quadrant_color(q):
    if q == 1:
         return "green"
    elif q == 2:
        return "yellow"
    elif q == 3:
        return "red"
    return "cyan"

@app.post("/detections/")
async def detect_teeth(file: UploadFile = File(...)):
    """
    Example endpoint that:
      - Reads the incoming X-ray image
      - Runs FasterRCNN
      - Applies a score filter + optional Unique-Class NMS
      - Decodes class indices to FDI numbers using label_mapper.decode
      - Returns the final bounding boxes + scores + FDI labels + PNG image
    """
    print("Processing image...")
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # adding clache for better visualization
    np_img = np.array(image)
    gray_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray_img)  # shape (H, W)
    clahe_pil = Image.fromarray(clahe_img).convert("RGB")

    image_tensor = torch.Tensor(np.array(image)).permute(2, 0, 1) / 255.0
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        prediction = model([image_tensor])[0]

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

    decoded_fdi_labels = label_mapper.decode(final_labels)
    predicted_quadrants = [int((label - 1) / 10 + 1) for label in decoded_fdi_labels]
    predicted_teeth = [int((label - 1) % 10 + 1) for label in decoded_fdi_labels]
    decoded_fdi_predicted_labels = [quadrant  * 10 + tooth for quadrant, tooth in zip(predicted_quadrants, predicted_teeth)]

    print("FDI predicitons", decoded_fdi_predicted_labels)

    # 7) Draw bounding boxes on a copy of the original image
    draw_image = clahe_pil.copy()
    image_height = draw_image.height
    draw = ImageDraw.Draw(draw_image)
    font=ImageFont.truetype("arial.ttf", 26)

    for box, pQuad, pTooth in zip(final_boxes, predicted_quadrants, predicted_teeth):
        x_min, y_min, x_max, y_max = box

        color = quadrant_color(pQuad)

        draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=6)
        if pQuad in [1, 2]:
            text_y = max(0, y_min - 60)
        else:
            # For lower quadrants, place text below
            text_y = min(y_max + 5, image_height - 30)
        
        draw.text((x_min, text_y), f"Q={pQuad}\nN={pTooth}", fill=color, font=font)

    # 8) Convert the updated image to PNG
    buffered = io.BytesIO()
    draw_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    # 9) Return the final results
    return {
        "processed_image": f"data:image/png;base64,{img_str}",
        "detections": {
            "boxes": final_boxes.tolist(),
            "labels": decoded_fdi_predicted_labels,
        }
    }


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
