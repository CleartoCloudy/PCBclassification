"""
PyTorch Dataset 类 - 加载PCB分类数据集
Author: CleartoCloudy
"""
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CLASS_NAMES, IMG_SIZE_DL


def get_train_transforms():
    """训练数据增强（含在线增强）"""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_val_transforms():
    """验证/测试数据变换（无增强）"""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


class PCBDataset(Dataset):
    """PCB缺陷分类数据集"""

    def __init__(self, root_dir, transform=None):
        """
        root_dir: 包含类别子文件夹的目录 (如 train/clahe/)
        transform: torchvision transforms
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        for class_id, cls_name in enumerate(CLASS_NAMES):
            class_dir = os.path.join(root_dir, cls_name)
            if not os.path.exists(class_dir):
                continue
            for img_name in os.listdir(class_dir):
                if img_name.endswith((".jpg", ".png", ".bmp")):
                    self.samples.append({
                        "path": os.path.join(class_dir, img_name),
                        "label": class_id,
                        "class_name": cls_name,
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = cv2.imread(sample["path"])
        if image is None:
            raise FileNotFoundError(f"无法读取图像: {sample['path']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMG_SIZE_DL, IMG_SIZE_DL))

        if self.transform:
            image = self.transform(image)

        return image, sample["label"]


def create_dataloaders(train_dir, val_dir, batch_size=32, num_workers=0):
    """创建训练和验证DataLoader"""
    train_dataset = PCBDataset(train_dir, transform=get_train_transforms())
    val_dataset = PCBDataset(val_dir, transform=get_val_transforms())

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    print(f"训练集: {len(train_dataset)} 张, 验证集: {len(val_dataset)} 张")
    return train_loader, val_loader, len(train_dataset), len(val_dataset)


def create_test_loader(test_dir, batch_size=32, num_workers=0):
    """创建测试DataLoader"""
    test_dataset = PCBDataset(test_dir, transform=get_val_transforms())
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    print(f"测试集: {len(test_dataset)} 张")
    return test_loader, len(test_dataset)
