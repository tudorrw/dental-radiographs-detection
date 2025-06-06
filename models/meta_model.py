import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import csv
import json
import torch
import numpy as np
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision import transforms as T
from tqdm import tqdm
# Make sure these imports point to the right modules
from utils.wbf import weighted_boxes_fusion
from utils.get_models import FineTunedModels
from torch.utils.data import DataLoader
from data.rcnn_teeth_dataset import TeethDataset
from utils.nms import UniqueClassNMSProcessor, CombinedNMS  # If you're using class-wise NMS
import models.dino.datasets.transforms as DT

device = "cuda" if torch.cuda.is_available() else "cpu"

models = FineTunedModels(device)

rcnn_model = models.get_faster_rcnn_model()
# yolo_model = models.get_yolo_model()
retinanet_model = models.get_retinanet_model()
dino_model, postprocessors = models.get_dino_model()


def normalize_boxes(boxes, w, h):
    """
    Convert pixel coords [x1,y1,x2,y2] → normalized [0..1].
    """
    boxes = boxes.astype(np.float32)
    boxes[:, [0,2]] /= w  # x1, x2
    boxes[:, [1,3]] /= h  # y1, y2
    return boxes


def denormalize_boxes(boxes, w, h):
    """
    Convert normalized coords [0..1] → pixel coords [x1,y1,x2,y2].
    """
    boxes = boxes.astype(np.float32)
    boxes[:, [0,2]] *= w
    boxes[:, [1,3]] *= h
    return boxes

@torch.no_grad()
def ensemble_predict(rcnn_model, retinanet_model, yolo_model, dino_model, postprocessors, dataloader):
    
    map_metric = MeanAveragePrecision(box_format="xyxy")
    
    dino_transforms = DT.Compose([
        DT.RandomResize([800], max_size=1333),
        DT.ToTensor(),
        DT.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    for batch in tqdm(dataloader):
        images = batch['image']
        targets = batch['targets']
        image_ids = batch['id']

        batch_size = len(images)
        for i in range(batch_size):
            img_tensor = images[i].to(device)  # shape [3, H, W]
            # We assume the dataset stores the original width/height in targets[i]
            w = targets[i]["width"].item() if "width" in targets[i] else img_tensor.shape[2]
            h = targets[i]["height"].item() if "height" in targets[i] else img_tensor.shape[1]

            # (A) Faster R-CNN
            rcnn_pred = rcnn_model([img_tensor])[0]
            rcnn_boxes  = rcnn_pred["boxes"].cpu().numpy()
            rcnn_scores = rcnn_pred["scores"].cpu().numpy()
            rcnn_labels = rcnn_pred["labels"].cpu().numpy()

            # (B) RetinaNet
            retina_pred = retinanet_model([img_tensor])[0]
            ret_boxes  = retina_pred["boxes"].cpu().numpy()
            ret_scores = retina_pred["scores"].cpu().numpy()
            ret_labels = retina_pred["labels"].cpu().numpy()

            
           # (C) DINO - CORRECTED
            img_pil = T.ToPILImage()(images[i])
            if img_pil.mode != "RGB":
                img_pil = img_pil.convert("RGB")

            img_tensor_dino, _ = dino_transforms(img_pil, None) 
            if device == "cuda":
                img_tensor_dino = img_tensor_dino.cuda()
            
            # Get resized dimensions
            dino_pred = dino_model(img_tensor_dino.unsqueeze(0))
            scale = torch.tensor([[h, w]])
            if device == "cuda":
                scale = scale.cuda()
            
            output = postprocessors["bbox"](dino_pred, scale)[0]

            dino_scores = output["scores"].cpu().numpy()
            dino_labels = (output["labels"] + 1).cpu().numpy()
            dino_boxes = output["boxes"].cpu().numpy()

            
            
            # if len(dino_boxes_resized) > 0:
            #     # Scale from resized back to original
            #     scale_x = w / resized_w
            #     scale_y = h / resized_h
                
            #     dino_boxes = dino_boxes_resized.copy()
            #     dino_boxes[:, [0, 2]] *= scale_x  # x1, x2
            #     dino_boxes[:, [1, 3]] *= scale_y  # y1, y2
                
            #     # Normalize for WBF
            #     dino_boxes_norm = dino_boxes.copy()
            #     dino_boxes_norm[:, [0, 2]] /= w
            #     dino_boxes_norm[:, [1, 3]] /= h
            # else:
            #     dino_boxes_norm = np.empty((0, 4))


            # Convert image for yolo
            # img_for_yolo = img_tensor.cpu().numpy()  # [C, H, W]
            # img_for_yolo = np.transpose(img_for_yolo, (1, 2, 0))  # [H, W, C]
            # img_for_yolo = (img_for_yolo * 255).astype(np.uint8)
            
            # # Ensure image has 3 channels (YOLO expects RGB)
            # if img_for_yolo.shape[-1] == 1:
            #     img_for_yolo = np.repeat(img_for_yolo, 3, axis=-1)
            
            # yolo_out = yolo_model.predict(img_for_yolo, conf=0.5)[0]
            # yolo_boxes  = yolo_out.boxes.xyxy.cpu().numpy()
            # yolo_scores = yolo_out.boxes.conf.cpu().numpy()
            # yolo_labels = yolo_out.boxes.cls.cpu().numpy()

            rcnn_boxes_norm = normalize_boxes(rcnn_boxes, w, h) if len(rcnn_boxes) > 0 else np.empty((0,4))
            ret_boxes_norm  = normalize_boxes(ret_boxes,  w, h) if len(ret_boxes)  > 0 else np.empty((0,4))
            dino_boxes_norm = normalize_boxes(dino_boxes, w, h) if len(dino_boxes) > 0 else np.empty((0,4))
            
            
            # yolo_boxes_norm = normalize_boxes(yolo_boxes, w, h) if len(yolo_boxes) > 0 else np.empty((0,4))

            boxes_list  = [rcnn_boxes_norm, ret_boxes_norm, dino_boxes_norm]
            scores_list = [rcnn_scores, ret_scores, dino_scores]
            labels_list = [rcnn_labels, ret_labels, dino_labels]
            
            # boxes_list = [dino_boxes_norm]
            # scores_list = [dino_scores]
            # labels_list = [dino_labels]
    
            if sum([len(b) for b in boxes_list]) == 0:
                # Update metric with empty pred
                final_target = [{
                    "boxes": targets[i]["boxes"],
                    "labels": targets[i]["labels"]
                }]
                map_metric.update(preds=[{"boxes": torch.empty((0,4)), "labels": torch.empty((0,), dtype=torch.long), "scores": torch.empty((0,))}],
                                  target=final_target)
                continue

            # (E) Weighted Boxes Fusion
            fused_boxes_norm, fused_scores, fused_labels = weighted_boxes_fusion(
                boxes_list,
                scores_list,
                labels_list,
                # weights=[7.5, 6], # or e.g. [2,1,1] if you trust RCNN more
                weights=[1.15,1.5,2.4],
                # weights=[1.15, 2],
                iou_thr=0.6,
                conf_type="avg",
            )

            # Denormalize back to pixels
            fused_boxes = denormalize_boxes(fused_boxes_norm, w, h)

            # (F) Convert to PyTorch Tensors so we can feed them to map_metric
            fused_boxes_t  = torch.tensor(fused_boxes, dtype=torch.float32)
            fused_scores_t = torch.tensor(fused_scores, dtype=torch.float32)
            fused_labels_t = torch.tensor(fused_labels, dtype=torch.long)

            final_pred = [{
                "boxes":  fused_boxes_t,
                "labels": fused_labels_t,
                "scores": fused_scores_t
            }]

            # Ground truth
            gt_boxes  = targets[i]["boxes"]
            gt_labels = targets[i]["labels"]

            final_target = [{
                "boxes": gt_boxes,      # already a Tensor in your dataset
                "labels": gt_labels
            }]

            # (G) Update mAP metric
            map_metric.update(preds=final_pred, target=final_target)

    # 3) After the loop, compute final metrics
    computed = map_metric.compute()


    keys_to_show = ["map", "map_50", "map_75", "map_large", "map_medium"]
    print("┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩")
    for k in keys_to_show:
        val = computed.get(k, None)
        if val is None:
            continue
        print(f"│ {k:<24} │ {val:>26} │")
    print("└───────────────────────────┴───────────────────────────┘")



if __name__ == '__main__':
    test_dataset = TeethDataset(
    image_dir="dataset/origin/quadrant_enumeration/xrays", 
    csv_path=f"dataset/pascal_voc/quadrant_enumeration/quadrant_enumeration_voc_test.csv",
    dataset_type="test"
    )

    test_loader = DataLoader(test_dataset, batch_size=2, num_workers=0, pin_memory=True, collate_fn=TeethDataset.collate_fn)
    ensemble_predict(rcnn_model, retinanet_model, None, dino_model, postprocessors, test_loader)

