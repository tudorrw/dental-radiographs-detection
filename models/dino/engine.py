# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""

import math
import os
import sys
from typing import Iterable

from utils.utils import slprint, to_device
from utils.box_ops import box_cxcywh_to_xyxy

import torch

import utils.misc as utils
from .datasets.coco_eval import CocoEvaluator
from .datasets.panoptic_eval import PanopticEvaluator
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.ops import box_iou
import seaborn as sns


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0,
    wo_class_error=False,
    lr_scheduler=None,
    args=None,
    logger=None,
    ema_m=None,
):
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    if not wo_class_error:
        metric_logger.add_meter("class_error", utils.SmoothedValue(window_size=1, fmt="{value:.2f}"))
    header = f"Epoch {epoch+1}"
    print_freq = 10

    _cnt = 0
    # Use tqdm for a clean progress bar
    is_main = utils.is_main_process()
    data_iter = data_loader
    if is_main:
        data_iter = tqdm(data_loader, desc=header, leave=True, ncols=100)
    for samples, targets in data_iter:
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=args.amp):
            if need_tgt_for_training:
                outputs = model(samples, targets)
            else:
                outputs = model(samples)

            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict

            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f"{k}_unscaled": v for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # amp backward function
        if args.amp:
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            # original backward function
            optimizer.zero_grad()
            losses.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        if args.onecyclelr:
            lr_scheduler.step()
        if args.use_ema:
            if epoch >= args.ema_epoch:
                ema_m.update(model)

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        if "class_error" in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced["class_error"])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # Update tqdm bar with current loss
        if is_main:
            data_iter.set_postfix({'loss': f'{loss_value:.4f}'})

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!" * 5)
                break

    if getattr(criterion, "loss_weight_decay", False):
        criterion.loss_weight_decay(epoch=epoch)
    if getattr(criterion, "tuning_matching", False):
        criterion.tuning_matching(epoch)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    if is_main:
        print("Averaged stats:", metric_logger)
    resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if getattr(criterion, "loss_weight_decay", False):
        resstat.update({f"weight_{k}": v for k, v in criterion.weight_dict.items()})
    return resstat


def denormalize_boxes(boxes, w, h):
    """
    Convert normalized coords [0..1] → pixel coords in COCO format [x,y,width,height].
    Input boxes are in format [x1,y1,x2,y2] normalized to [0,1]
    """
    boxes = boxes.float()
    # Convert from normalized [x1,y1,x2,y2] to COCO format [x,y,width,height]
    x1 = boxes[:, 0] * w
    y1 = boxes[:, 1] * h
    x2 = boxes[:, 2] * w
    y2 = boxes[:, 3] * h
    
    # Convert to COCO format
    boxes[:, 0] = x1  # x
    boxes[:, 1] = y1  # y
    boxes[:, 2] = x2 - x1  # width
    boxes[:, 3] = y2 - y1  # height
    return boxes


@torch.no_grad()
def evaluate(
    model,
    criterion,
    postprocessors,
    data_loader,
    base_ds,
    device,
    output_dir,
    wo_class_error=False,
    args=None,
    logger=None,
    epoch=None,
):
    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    if not wo_class_error:
        metric_logger.add_meter("class_error", utils.SmoothedValue(window_size=1, fmt="{value:.2f}"))
    header = "Evaluate:"

    iou_types = tuple(k for k in ("segm", "bbox") if k in postprocessors.keys())
    useCats = True
    try:
        useCats = args.useCats
    except:
        useCats = True
    if not useCats:
        print("useCats: {} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!".format(useCats))
    coco_evaluator = CocoEvaluator(base_ds, iou_types, useCats=useCats)

    panoptic_evaluator = None
    if "panoptic" in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    _cnt = 0
    output_state_dict = {}  # for debug only
    # For confusion matrix
    all_gt_labels = []
    all_pred_labels = []
    # Use tqdm for a clean progress bar
    is_main = utils.is_main_process()
    data_iter = data_loader
    if is_main:
        data_iter = tqdm(data_loader, desc=header, leave=True, ncols=100)
    for samples, targets in data_iter:
        samples = samples.to(device)
        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=args.amp):
            if need_tgt_for_training:
                outputs = model(samples, targets)
            else:
                outputs = model(samples)
            loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f"{k}_unscaled": v for k, v in loss_dict_reduced.items()}
        metric_logger.update(
            loss=sum(loss_dict_reduced_scaled.values()), **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled
        )
        if "class_error" in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced["class_error"])

        # Update tqdm bar with current loss
        if is_main:
            loss_value = sum(loss_dict_reduced_scaled.values()).item() if loss_dict_reduced_scaled else 0.0
            data_iter.set_postfix({'loss': f'{loss_value:.4f}'})

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        results = postprocessors["bbox"](outputs, orig_target_sizes)
        # print("results: ", results)
        # [scores: [100], labels: [100], boxes: [100, 4]] x B
        if "segm" in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors["segm"](results, outputs, orig_target_sizes, target_sizes)
        res = {target["image_id"].item(): output for target, output in zip(targets, results)}

        # Collect ground truth and predicted labels for confusion matrix
        for target, output in zip(targets, results):
            # Get predictions and targets
            pred_boxes = output["boxes"]
            pred_scores = output["scores"]
            pred_labels = output["labels"]
            
            # First denormalize the target boxes
            denorm_boxes = target["boxes"].clone()
            denorm_boxes[:, [0,2]] *= orig_target_sizes[0][1]  # scale x coordinates by width
            denorm_boxes[:, [1,3]] *= orig_target_sizes[0][0]  # scale y coordinates by height
            
            print("pred_boxes: ", pred_boxes)
            print("pred_labels: ", pred_labels)
            print("pred_scores: ", pred_scores)
            # Then convert to xyxy format
            true_boxes = box_cxcywh_to_xyxy(denorm_boxes)
            # Adjust labels: add 1 to make teeth 1-32 (from 0-31)
            true_labels = target["labels"]
            
            # Skip if no ground truth or predictions
            if len(true_boxes) == 0 or len(pred_boxes) == 0:
                continue
                
            # Filter predictions by score threshold
            score_threshold = 0.3
            keep_indices = torch.where(pred_scores > score_threshold)[0]
            if len(keep_indices) == 0:
                continue
            pred_boxes = pred_boxes[keep_indices]
            pred_labels = pred_labels[keep_indices]
            
            # Calculate IoU between all pred and gt boxes
            print("pred_boxes: ", pred_boxes)
            print("true_boxes: ", true_boxes)
            ious = box_iou(pred_boxes, true_boxes)
            print("ious: ", ious)
            
            # For each ground truth, find best matching prediction
            for gt_idx in range(len(true_labels)):
                gt_label = true_labels[gt_idx].item()
                all_gt_labels.append(gt_label)
                
                if len(ious) == 0:  # No predictions for this image
                    # all_pred_labels.append(-1)  # Background
                    continue
                
                # Find best prediction match
                best_iou, best_idx = torch.max(ious[:, gt_idx], dim=0)
                
                # If IoU is high enough, consider it a match
                if best_iou > 0.5:
                    all_pred_labels.append(pred_labels[best_idx].item())  # Add 1 to make teeth 1-32
                    
                    # Remove this prediction to avoid double matching
                    mask = torch.ones(ious.shape[0], dtype=torch.bool, device=ious.device)
                    mask[best_idx] = False
                    ious = ious[mask]
                    pred_labels = pred_labels[mask]
                else:
                    # No match with high enough IoU
                    all_pred_labels.append(-1)  # Consider as background

        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

        if args.save_results:
            for i, (tgt, res, outbbox) in enumerate(zip(targets, results, outputs["pred_boxes"])):
                gt_bbox = tgt["boxes"]
                gt_label = tgt["labels"]
                gt_info = torch.cat((gt_bbox, gt_label.unsqueeze(-1)), 1)
                _res_bbox = outbbox
                _res_prob = res["scores"]
                _res_label = res["labels"]
                res_info = torch.cat((_res_bbox, _res_prob.unsqueeze(-1), _res_label.unsqueeze(-1)), 1)
                if "gt_info" not in output_state_dict:
                    output_state_dict["gt_info"] = []
                output_state_dict["gt_info"].append(gt_info.cpu())
                if "res_info" not in output_state_dict:
                    output_state_dict["res_info"] = []
                output_state_dict["res_info"].append(res_info.cpu())

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!" * 5)
                break

    if args.save_results:
        import os.path as osp
        savepath = osp.join(args.output_dir, "results-{}.pkl".format(utils.get_rank()))
        print("Saving res to {}".format(savepath))
        torch.save(output_state_dict, savepath)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    if is_main:
        print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if coco_evaluator is not None:
        if "bbox" in postprocessors.keys():
            stats["coco_eval_bbox"] = coco_evaluator.coco_eval["bbox"].stats.tolist()
        if "segm" in postprocessors.keys():
            stats["coco_eval_masks"] = coco_evaluator.coco_eval["segm"].stats.tolist()
    if panoptic_res is not None:
        stats["PQ_all"] = panoptic_res["All"]
        stats["PQ_th"] = panoptic_res["Things"]
        stats["PQ_st"] = panoptic_res["Stuff"]

    # Compute and save confusion matrix
    if len(all_gt_labels) > 0 and len(all_pred_labels) > 0:
        # Convert lists to numpy arrays
        y_true = np.array(all_gt_labels)
        y_pred = np.array(all_pred_labels)
        
        # Ensure both arrays have the same length by padding with -1 (background)
        max_len = max(len(y_true), len(y_pred))
        y_true_padded = np.full(max_len, -1)
        y_pred_padded = np.full(max_len, -1)
        y_true_padded[:len(y_true)] = y_true
        y_pred_padded[:len(y_pred)] = y_pred
        
        # Get unique classes (including background -1)
        classes = np.arange(-1, 32)  # -1 for background, 0-31 for teeth
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true_padded, y_pred_padded, labels=classes)
        
        # Plot confusion matrix
        plt.figure(figsize=(20, 18))
        
        # Normalize confusion matrix for better visualization
        row_sums = cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.divide(cm.astype('float'), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
        cm_norm = np.nan_to_num(cm_norm)  # Replace NaN with 0
        
        # Create heatmap
        sns.heatmap(
            cm_norm,
            annot=True,
            cmap="Blues",
            fmt='.2f',
            square=True,
            xticklabels=[f"Class {i}" for i in classes],
            yticklabels=[f"Class {i}" for i in classes]
        )
        
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Normalized Confusion Matrix')
        
        # Save figure
        plt.savefig(os.path.join(output_dir, f"confusion_matrix_epoch_{epoch}.png"))
        plt.close()

    return stats, coco_evaluator


@torch.no_grad()
def test(
    model,
    criterion,
    postprocessors,
    data_loader,
    base_ds,
    device,
    output_dir,
    wo_class_error=False,
    args=None,
    logger=None,
):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    # if not wo_class_error:
    #     metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = "Test:"

    iou_types = tuple(k for k in ("segm", "bbox") if k in postprocessors.keys())
    # coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if "panoptic" in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    final_res = []
    for samples, targets in metric_logger.log_every(data_loader, 10, header, logger=logger):
        samples = samples.to(device)

        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        # loss_dict = criterion(outputs, targets)
        # weight_dict = criterion.weight_dict

        # # reduce losses over all GPUs for logging purposes
        # loss_dict_reduced = utils.reduce_dict(loss_dict)
        # loss_dict_reduced_scaled = {k: v * weight_dict[k]
        #                             for k, v in loss_dict_reduced.items() if k in weight_dict}
        # loss_dict_reduced_unscaled = {f'{k}_unscaled': v
        #                               for k, v in loss_dict_reduced.items()}
        # metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
        #                      **loss_dict_reduced_scaled,
        #                      **loss_dict_reduced_unscaled)
        # if 'class_error' in loss_dict_reduced:
        #     metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors["bbox"](outputs, orig_target_sizes, not_to_xyxy=True)
        # [scores: [100], labels: [100], boxes: [100, 4]] x B
        if "segm" in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors["segm"](results, outputs, orig_target_sizes, target_sizes)
        res = {target["image_id"].item(): output for target, output in zip(targets, results)}
        for image_id, outputs in res.items():
            _scores = outputs["scores"].tolist()
            _labels = outputs["labels"].tolist()
            _boxes = outputs["boxes"].tolist()
            for s, l, b in zip(_scores, _labels, _boxes):
                assert isinstance(l, int)
                itemdict = {
                    "image_id": int(image_id),
                    "category_id": l,
                    "bbox": b,
                    "score": s,
                }
                final_res.append(itemdict)

    if args.output_dir:
        import json

        with open(args.output_dir + f"/results{args.rank}.json", "w") as f:
            json.dump(final_res, f)

    return final_res
