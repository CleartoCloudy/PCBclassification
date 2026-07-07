"""
ResNet18 分类模型（支持SE/CBAM注意力机制）
Author: CleartoCloudy
"""
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

from deep_learning.models.attention import SELayer, CBAM


def create_resnet18(num_classes=6, pretrained=True, attention=None):
    """
    创建ResNet18模型

    参数:
        num_classes: 分类类别数
        pretrained: 是否使用预训练权重
        attention: None | "se" | "cbam"  注意力机制类型

    返回: model
    """
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)

    if attention is None:
        # 基础ResNet18：修改全连接层
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif attention == "se":
        # 插入SE模块到每个BasicBlock之后
        _insert_se_to_resnet(model, num_classes)

    elif attention == "cbam":
        # 插入CBAM模块到每个BasicBlock之后
        _insert_cbam_to_resnet(model, num_classes)

    return model


def _insert_se_to_resnet(model, num_classes):
    """在ResNet18的每个BasicBlock后插入SE模块"""
    _replace_fc(model, num_classes)
    se_channels = [64, 64, 128, 128, 256, 256, 512, 512]
    idx = 0

    def _insert_se(block, name=""):
        nonlocal idx
        for child_name, child in block.named_children():
            if isinstance(child, nn.Sequential):
                _insert_se(child, f"{name}.{child_name}" if name else child_name)
            elif child.__class__.__name__ == "BasicBlock":
                _insert_se(child, f"{name}.{child_name}" if name else child_name)
        if block.__class__.__name__ == "BasicBlock" and idx < len(se_channels):
            block.add_module("se", SELayer(se_channels[idx]))
            idx += 1

    _insert_se(model)


def _insert_cbam_to_resnet(model, num_classes):
    """在ResNet18的每个BasicBlock后插入CBAM模块"""
    _replace_fc(model, num_classes)
    cbam_channels = [64, 64, 128, 128, 256, 256, 512, 512]
    idx = 0

    def _insert_cbam(block, name=""):
        nonlocal idx
        for child_name, child in block.named_children():
            if isinstance(child, nn.Sequential):
                _insert_cbam(child, f"{name}.{child_name}" if name else child_name)
            elif child.__class__.__name__ == "BasicBlock":
                _insert_cbam(child, f"{name}.{child_name}" if name else child_name)
        if block.__class__.__name__ == "BasicBlock" and idx < len(cbam_channels):
            block.add_module("cbam", CBAM(cbam_channels[idx]))
            idx += 1

    _insert_cbam(model)


def _replace_fc(model, num_classes):
    """替换全连接层"""
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)


def count_parameters(model):
    """统计模型参数量（百万）"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
