# ------------------------------------------------------------------------
# LW-DETR
# Copyright (c) 2024 Baidu. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path
import os
import random
import torch
import torch.utils.data
import torchvision

import datasets.transforms as T


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCoco()

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {"image_id": image_id, "annotations": target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


class ConvertCoco(object):

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if "iscrowd" not in obj or obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["image_id"] = image_id

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor(
            [obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno]
        )
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set):

    normalize = T.Compose(
        [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )

    max_size = 224
    # scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    scales = [224]

    if image_set == "train":
        return T.Compose(
            [
                T.RandomHorizontalFlip(),
                T.RandomResize(scales, max_size=max_size),
                # T.RandomSelect(
                # T.RandomResize(scales, max_size=max_size),
                # T.Compose(
                #     [
                #         # T.RandomResize([400, 500, 600]),
                #         # T.RandomSizeCrop(384, 600),
                #         T.RandomResize(scales, max_size=max_size),
                #     ]
                # ),
                # ),
                normalize,
            ]
        )

    if image_set == "val":
        return T.Compose(
            [
                T.RandomResize([224], max_size=max_size),
                normalize,
            ]
        )
    if image_set == "val_speed":
        return T.Compose(
            [
                T.SquareResize([max_size]),
                normalize,
            ]
        )

    raise ValueError(f"unknown {image_set}")


def make_coco_transforms_square_div_64(image_set):
    """ """

    normalize = T.Compose(
        [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )

    scales = [448, 512, 576, 640, 704, 768, 832, 896]

    if image_set == "train":
        return T.Compose(
            [
                T.RandomHorizontalFlip(),
                T.RandomSelect(
                    T.SquareResize(scales),
                    T.Compose(
                        [
                            T.RandomResize([400, 500, 600]),
                            T.RandomSizeCrop(384, 600),
                            T.SquareResize(scales),
                        ]
                    ),
                ),
                normalize,
            ]
        )

    if image_set == "val":
        return T.Compose(
            [
                T.SquareResize([640]),
                normalize,
            ]
        )
    if image_set == "val_speed":
        return T.Compose(
            [
                T.SquareResize([640]),
                normalize,
            ]
        )

    raise ValueError(f"unknown {image_set}")


def build_coco(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f"provided COCO path {root} does not exist"
    mode = "instances"
    PATHS = {
        "train": (root / "train2017", root / "annotations" / f"{mode}_train2017.json"),
        "val": (root / "val2017", root / "annotations" / f"{mode}_val2017.json"),
        "test": (
            root / "test2017",
            root / "annotations" / f"image_info_test-dev2017.json",
        ),
    }

    img_folder, ann_file = PATHS[image_set.split("_")[0]]

    try:
        square_resize = args.square_resize
    except:
        square_resize = False

    try:
        square_resize_div_64 = args.square_resize_div_64
    except:
        square_resize_div_64 = False

    if square_resize_div_64:
        dataset = CocoDetection(
            img_folder,
            ann_file,
            transforms=make_coco_transforms_square_div_64(image_set),
        )
    else:
        dataset = CocoDetection(
            img_folder, ann_file, transforms=make_coco_transforms(image_set)
        )
    return dataset


def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f"provided COCO path {root} does not exist"
    PATHS = {
        "train": (root / "train", root / f"train.json"),
        "val": (root / "val", root / f"val.json"),
    }

    img_folder, ann_file = PATHS[image_set.split("_")[0]]

    try:
        square_resize = args.square_resize
    except:
        square_resize = False

    try:
        square_resize_div_64 = args.square_resize_div_64
    except:
        square_resize_div_64 = False

    if square_resize_div_64:
        dataset = CocoDetection(
            img_folder,
            ann_file,
            transforms=make_coco_transforms_square_div_64(image_set),
        )
    else:
        dataset = CocoDetection(
            img_folder, ann_file, transforms=make_coco_transforms(image_set)
        )
    return dataset


def build_doclaynet(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f"provided COCO path {root} does not exist"
    PATHS = {
        "train": (root / "PNG", root / f"COCO/train.json"),
        "val": (root / "PNG", root / f"COCO/val.json"),
    }

    img_folder, ann_file = PATHS[image_set.split("_")[0]]

    try:
        square_resize = args.square_resize
    except:
        square_resize = False

    try:
        square_resize_div_64 = args.square_resize_div_64
    except:
        square_resize_div_64 = False

    if square_resize_div_64:
        dataset = CocoDetection(
            img_folder,
            ann_file,
            transforms=make_coco_transforms_square_div_64(image_set),
        )
    else:
        dataset = CocoDetection(
            img_folder, ann_file, transforms=make_coco_transforms(image_set)
        )
    return dataset
