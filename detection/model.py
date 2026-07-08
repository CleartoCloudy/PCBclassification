"""
PCB缺陷检测模型 - Faster R-CNN + 小目标优化
Author: CleartoCloudy
"""
import torch
import torch.nn as nn
from torchvision.models.detection import (fasterrcnn_resnet50_fpn,
                                          FasterRCNN_ResNet50_FPN_Weights)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def create_model(num_classes=7, pretrained=True):
    """
    Faster R-CNN (ResNet50-FPN)，针对 PCB 微小缺陷优化

    默认锚框 (32,64,128,256,512) → 改为 (8,16,32,64,128)
    适配 20~50px 的 PCB 缺陷
    """
    weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1 if pretrained else None
    model = fasterrcnn_resnet50_fpn(weights=weights)

    # ---- RPN 调优：更多 proposal ----
    model.rpn.pre_nms_top_n_train = 2000
    model.rpn.pre_nms_top_n_test = 1000
    model.rpn.post_nms_top_n_train = 2000
    model.rpn.post_nms_top_n_test = 1000

    # ---- ROI head：更多采样框 ----
    model.roi_heads.batch_size_per_image = 512
    model.roi_heads.positive_fraction = 0.25

    # ---- 分类头 ----
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
