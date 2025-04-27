# ------------------------------------------------------------------------
# LW-DETR
# Copyright (c) 2024 Baidu. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
COCO evaluator that works in distributed mode.

Mostly copy-paste from https://github.com/pytorch/vision/blob/edfd5a7/references/detection/coco_eval.py
The difference is that there is less copy-pasting from pycocotools
in the end of the file, as python3 can suppress prints with contextlib
"""
import os
import contextlib
import copy
import numpy as np
import torch

from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import pycocotools.mask as mask_util

from util.misc import all_gather


class CocoEvaluator(object):
    def __init__(self, coco_gt, iou_types):
        assert isinstance(iou_types, (list, tuple))
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt

        self.iou_types = iou_types
        self.coco_eval = {}
        for iou_type in iou_types:
            self.coco_eval[iou_type] = COCOeval(coco_gt, iouType=iou_type)

        self.img_ids = []
        self.eval_imgs = {k: [] for k in iou_types}

    def get_metrics(self):
        """
        Calculate detection metrics using a similar approach to Ultralytics YOLOv8.
        Returns precision, recall, mAP50, and mAP50-95 metrics.
        """
        if (
            not hasattr(self.coco_eval["bbox"], "eval")
            or self.coco_eval["bbox"].eval is None
        ):
            return {
                "precision": 0.0,
                "recall": 0.0,
                "mAP50": 0.0,
                "mAP5095": 0.0,
                "fitness": 0.0,
            }

        coco_eval = self.coco_eval["bbox"]

        # Get precision and recall arrays from COCO evaluation
        precision = coco_eval.eval[
            "precision"
        ]  # shape [TxRxKxAxM] where T=10 IoU thresholds, R=recall, K=classes, A=areas, M=max dets
        recall = coco_eval.eval[
            "recall"
        ]  # shape [TxKxA] where T=10 IoU thresholds, K=classes, A=areas

        # Get number of classes and create metric arrays
        num_classes = precision.shape[2]
        p_per_class = np.zeros(num_classes)
        r_per_class = np.zeros(num_classes)
        f1_per_class = np.zeros(num_classes)
        ap50_per_class = np.zeros(num_classes)
        ap_per_class = np.zeros(num_classes)

        # x-coordinates for curve interpolation (similar to Ultralytics)
        x = np.linspace(0, 1, 101)  # 101-point interpolation (COCO standard)

        # Process each class
        for class_idx in range(num_classes):
            # For AP50 (IoU=0.5)
            iou_idx = 0  # IoU@0.5
            # Extract precision and recall curves for this class
            # Use area='all' (idx 0) and max detections (idx -1)
            prec_curve = precision[iou_idx, :, class_idx, 0, -1]
            rec_curve = np.linspace(
                0, 1, prec_curve.shape[0]
            )  # COCO standard recall points

            # Remove -1 values (similar to Ultralytics)
            valid_idx = prec_curve > -1
            prec_curve = prec_curve[valid_idx]
            rec_curve = rec_curve[valid_idx]

            if len(prec_curve) > 0:
                # Compute AP using Ultralytics method
                # Append sentinel values
                mrec = np.concatenate(([0.0], rec_curve, [1.0]))
                mpre = np.concatenate(([1.0], prec_curve, [0.0]))

                # Compute the precision envelope (maximum precision for each recall value)
                mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

                # Integrate using interpolation (COCO approach)
                ap50 = np.trapz(np.interp(x, mrec, mpre), x)
                ap50_per_class[class_idx] = ap50

                # Get precision and recall at max F1 point
                # Calculate F1 curve
                with np.errstate(divide="ignore", invalid="ignore"):
                    f1_curve = (
                        2 * (mpre[:-1] * mrec[:-1]) / (mpre[:-1] + mrec[:-1] + 1e-16)
                    )

                # Find point of maximum F1
                max_f1_idx = np.nanargmax(f1_curve)
                p_per_class[class_idx] = mpre[max_f1_idx]
                r_per_class[class_idx] = mrec[max_f1_idx]
                f1_per_class[class_idx] = f1_curve[max_f1_idx]

            # Calculate AP across all IoU thresholds (mAP@0.5:0.95)
            ap_all_ious = []
            for iou_idx in range(len(coco_eval.params.iouThrs)):
                prec_curve = precision[iou_idx, :, class_idx, 0, -1]
                valid_idx = prec_curve > -1
                prec_curve = prec_curve[valid_idx]

                if len(prec_curve) > 0:
                    rec_curve = np.linspace(0, 1, len(prec_curve))

                    # Compute AP for this IoU threshold
                    mrec = np.concatenate(([0.0], rec_curve, [1.0]))
                    mpre = np.concatenate(([1.0], prec_curve, [0.0]))
                    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
                    ap_iou = np.trapz(np.interp(x, mrec, mpre), x)
                    ap_all_ious.append(ap_iou)
                else:
                    ap_all_ious.append(0.0)

            ap_per_class[class_idx] = np.mean(ap_all_ious)

        # Calculate mean metrics
        mp = np.mean(p_per_class)
        mr = np.mean(r_per_class)
        mf1 = np.mean(f1_per_class)
        map50 = np.mean(ap50_per_class)
        map = np.mean(ap_per_class)

        # Calculate fitness score using Ultralytics weighting
        w = np.array([0.0, 0.0, 0.1, 0.9])  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
        fitness = (np.array([mp, mr, map50, map]) * w).sum()

        # Format return values
        metrics = {
            "precision": float(mp),
            "recall": float(mr),
            "f1": float(mf1),
            "mAP50": float(map50),
            "mAP5095": float(map),
            "fitness": float(fitness),
            # "precision_per_class": p_per_class,
            # "recall_per_class": r_per_class,
            # "f1_per_class": f1_per_class,
            # "ap50_per_class": ap50_per_class,
            # "ap_per_class": ap_per_class,
        }

        return metrics

    def update(self, predictions):
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)

            # suppress pycocotools prints
            with open(os.devnull, "w") as devnull:
                with contextlib.redirect_stdout(devnull):
                    coco_dt = COCO.loadRes(self.coco_gt, results) if results else COCO()
            coco_eval = self.coco_eval[iou_type]

            coco_eval.cocoDt = coco_dt
            coco_eval.params.imgIds = list(img_ids)
            img_ids, eval_imgs = evaluate(coco_eval)

            self.eval_imgs[iou_type].append(eval_imgs)

    def synchronize_between_processes(self):
        for iou_type in self.iou_types:
            self.eval_imgs[iou_type] = np.concatenate(self.eval_imgs[iou_type], 2)
            create_common_coco_eval(
                self.coco_eval[iou_type], self.img_ids, self.eval_imgs[iou_type]
            )

    def accumulate(self):
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    def summarize(self):
        for iou_type, coco_eval in self.coco_eval.items():
            print("IoU metric: {}".format(iou_type))
            coco_eval.summarize()

    def prepare(self, predictions, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        elif iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        elif iou_type == "keypoints":
            return self.prepare_for_coco_keypoint(predictions)
        else:
            raise ValueError("Unknown iou type {}".format(iou_type))

    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results

    def prepare_for_coco_segmentation(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            scores = prediction["scores"]
            labels = prediction["labels"]
            masks = prediction["masks"]

            masks = masks > 0.5

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            rles = [
                mask_util.encode(
                    np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F")
                )[0]
                for mask in masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "segmentation": rle,
                        "score": scores[k],
                    }
                    for k, rle in enumerate(rles)
                ]
            )
        return coco_results

    def prepare_for_coco_keypoint(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            keypoints = prediction["keypoints"]
            keypoints = keypoints.flatten(start_dim=1).tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "keypoints": keypoint,
                        "score": scores[k],
                    }
                    for k, keypoint in enumerate(keypoints)
                ]
            )
        return coco_results


def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def merge(img_ids, eval_imgs):
    all_img_ids = all_gather(img_ids)
    all_eval_imgs = all_gather(eval_imgs)

    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)

    merged_eval_imgs = []
    for p in all_eval_imgs:
        merged_eval_imgs.append(p)

    merged_img_ids = np.array(merged_img_ids)
    merged_eval_imgs = np.concatenate(merged_eval_imgs, 2)

    # keep only unique (and in sorted order) images
    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)
    merged_eval_imgs = merged_eval_imgs[..., idx]

    return merged_img_ids, merged_eval_imgs


def create_common_coco_eval(coco_eval, img_ids, eval_imgs):
    img_ids, eval_imgs = merge(img_ids, eval_imgs)
    img_ids = list(img_ids)
    eval_imgs = list(eval_imgs.flatten())

    coco_eval.evalImgs = eval_imgs
    coco_eval.params.imgIds = img_ids
    coco_eval._paramsEval = copy.deepcopy(coco_eval.params)


#################################################################
# From pycocotools, just removed the prints and fixed
# a Python3 bug about unicode not defined
#################################################################


def evaluate(self):
    """
    Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
    :return: None
    """
    # tic = time.time()
    # print('Running per image evaluation...')
    p = self.params
    # add backward compatibility if useSegm is specified in params
    if p.useSegm is not None:
        p.iouType = "segm" if p.useSegm == 1 else "bbox"
        print(
            "useSegm (deprecated) is not None. Running {} evaluation".format(p.iouType)
        )
    # print('Evaluate annotation type *{}*'.format(p.iouType))
    p.imgIds = list(np.unique(p.imgIds))
    if p.useCats:
        p.catIds = list(np.unique(p.catIds))
    p.maxDets = sorted(p.maxDets)
    self.params = p

    self._prepare()
    # loop through images, area range, max detection number
    catIds = p.catIds if p.useCats else [-1]

    if p.iouType == "segm" or p.iouType == "bbox":
        computeIoU = self.computeIoU
    elif p.iouType == "keypoints":
        computeIoU = self.computeOks
    self.ious = {
        (imgId, catId): computeIoU(imgId, catId)
        for imgId in p.imgIds
        for catId in catIds
    }

    evaluateImg = self.evaluateImg
    maxDet = p.maxDets[-1]
    evalImgs = [
        evaluateImg(imgId, catId, areaRng, maxDet)
        for catId in catIds
        for areaRng in p.areaRng
        for imgId in p.imgIds
    ]
    # this is NOT in the pycocotools code, but could be done outside
    evalImgs = np.asarray(evalImgs).reshape(len(catIds), len(p.areaRng), len(p.imgIds))
    self._paramsEval = copy.deepcopy(self.params)
    # toc = time.time()
    # print('DONE (t={:0.2f}s).'.format(toc-tic))
    return p.imgIds, evalImgs


#################################################################
# end of straight copy from pycocotools, just removing the prints
#################################################################
