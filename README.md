# TriFuse: a Feature Fusion approach for Chart Detection

## Method

### Backbone

We utilize the fusion feature of HiFuse architecture to fuse Global features and Local features for small image size (224):

- Global Block: utilize the parallel computing and temporal feature extracting of Transformer blocks to capture long distance dependencies, hence the name Global.
- Local Block: utilize the inductive bias of Convolution blocks to extract small features that are "invisible" to Transformer.
- HFF Block: fuse features map from Global and Local blocks.

### Head

End-to-end Object Detection has recently received its attention due to the emergence of DETR. Therefore, we hypothesize that by fusing the rich feature maps from each stage of HFF blocks (using C2f block from YOLOv8) combine with a light-weight variant of DETR, the result will be promising.

- We cut one stage of HiFuse making it contains only 3 HFF blocks, hence "TriFuse", then we utilize the architecture from LW-DETR which extracts feature maps at each stage from TriFuse and fuse them using C2f block from YOLOv8.
- The whole process is and End-to-End Detection, hence no hand-crafted anchor boxes is required, we use learnable queries instead (stated in DETR).
- The research is reaching final state, final results coming soon...

### Training

- To improve the overall outputs of the model, we implement the CAEv2 method to train the backbone first with the DocLayNet Dataset.
- Then we continue to fine-tune the whole model using DocLayNet to achieve a competitive result in the chart domain.

## Reference

- [HiFuse](https://arxiv.org/abs/2209.10218)
- [Vision Transformer](https://arxiv.org/abs/2010.11929)
- [Feature Pyramid Network](https://arxiv.org/abs/1612.03144)
- [ViT-UperNet](https://link.springer.com/article/10.1007/s40747-024-01359-6)
- [DETR](https://arxiv.org/abs/2005.12872)
- [LW-DETR](https://arxiv.org/abs/2406.03459)
- [CAEv2](https://openreview.net/forum?id=f36LaK7M0F)
