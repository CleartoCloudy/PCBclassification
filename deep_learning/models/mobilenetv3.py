"""
MobileNetV3 Small 轻量级分类模型
Author: CleartoCloudy
"""
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


def create_mobilenetv3(num_classes=6, pretrained=True):
    """
    创建MobileNetV3 Small模型

    参数:
        num_classes: 分类类别数
        pretrained: 是否使用预训练权重

    返回: model
    """
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = mobilenet_v3_small(weights=weights)

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)

    return model


def count_parameters(model):
    """统计模型参数量（百万）"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
