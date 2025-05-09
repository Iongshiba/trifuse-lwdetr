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

import torch.utils.data
import torchvision

from .coco import build_coco, build_doclaynet
from .coco import build as build_chartrec
from .o365 import build_o365


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco


def build_dataset(image_set, args):
    if args.dataset_file == "doclaynet":
        return build_doclaynet(image_set, args)
    if args.dataset_file == "coco":
        return build_coco(image_set, args)
    if args.dataset_file == "o365":
        return build_o365(image_set, args)
    if args.dataset_file == "chartrec":
        return build_chartrec(image_set, args)
    raise ValueError(f"dataset {args.dataset_file} not supported")
