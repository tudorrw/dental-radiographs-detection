import numpy as np
import cv2
import io
import base64
from PIL import Image, ImageDraw, ImageFont
from utils.mapper import ToothLabelMapper
import torch

async def read_convert_image(file):
    contents = await file.read()
    return Image.open(io.BytesIO(contents)).convert("RGB")


def clahe(image):
    np_img = np.array(image)
    gray_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray_img)  # shape (H, W)
    return Image.fromarray(clahe_img).convert("RGB")


def quadrant_color(q):
    if q == 1:
         return (0, 128, 0, 128)      # Green, 50% transparent
    elif q == 2:
         return (255, 255, 0, 128)    # Yellow, 50% transparent
    elif q == 3:
         return (255, 0, 0, 128)      # Red, 50% transparent
    return (0, 255, 255, 128)          # Cyan, 50% transpare


label_mapper = ToothLabelMapper()

def decode_teeth_numbers(labels): 
    decoded_fdi_labels = label_mapper.decode(labels)
    predicted_quadrants = [int((label - 1) / 10 + 1) for label in decoded_fdi_labels]
    predicted_teeth = [int((label - 1) % 10 + 1) for label in decoded_fdi_labels]
    decoded_fdi_predicted_labels = [quadrant  * 10 + tooth for quadrant, tooth in zip(predicted_quadrants, predicted_teeth)]

    return predicted_quadrants, predicted_teeth, decoded_fdi_predicted_labels


def draw_boxes(clahe_pil, final_boxes, predicted_quadrants, predicted_teeth):
     
    draw_image = clahe_pil.copy()
    image_height = draw_image.height
    draw = ImageDraw.Draw(draw_image)
    font=ImageFont.truetype("arial.ttf", 22)

    for box, pQuad, pTooth in zip(final_boxes, predicted_quadrants, predicted_teeth):
        x_min, y_min, x_max, y_max = box

        color = quadrant_color(pQuad)

        draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=3)
        if pQuad in [1, 2]:
            text_y = max(0, y_min - 60)
        else:
            # For lower quadrants, place text below
            text_y = min(y_max + 5, image_height - 30)
        
        draw.text((x_min, text_y), f"Q={pQuad}\nN={pTooth}", fill=color, font=font)

    # 8) Convert the updated image to PNG
    buffered = io.BytesIO()
    draw_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def output_json(image_str, boxes, scores, labels):
    return {
        "processed_image": f"data:image/png;base64,{image_str}",
        "detections": {
            "boxes": boxes.tolist(),
            "scores": scores.tolist(),
            "labels": labels
        }
    }