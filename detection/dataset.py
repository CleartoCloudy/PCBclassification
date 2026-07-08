"""
PCB缺陷检测数据集 - 保持大图 + bounding box 标注
Author: CleartoCloudy
"""
import os
import sys
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATASET_ORIGINAL, CLASS_NAMES, RANDOM_SEED

random.seed(RANDOM_SEED)

# 缺陷类型映射：标注中的 1~6 → 0~5
CLASS_MAP = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}


def find_pcbdata_root():
    """查找 PCBData 根目录"""
    candidates = [
        os.path.join(DATASET_ORIGINAL, "DeepPCB", "PCBData"),
        os.path.join(DATASET_ORIGINAL, "PCBData"),
        DATASET_ORIGINAL,
    ]
    for cand in candidates:
        if os.path.isdir(cand) and any(
            d.startswith("group") for d in os.listdir(cand)
            if os.path.isdir(os.path.join(cand, d))):
            return cand
    raise FileNotFoundError(f"未找到 PCBData 目录，请检查 {DATASET_ORIGINAL}")


def parse_split_file(split_path, pcbdata_root):
    """
    解析 trainval.txt 或 test.txt
    返回: [(image_path, annot_path), ...]
    """
    pairs = []
    with open(split_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            img_rel, ann_rel = parts[0], parts[1]
            img_full = os.path.join(pcbdata_root, img_rel)
            ann_full = os.path.join(pcbdata_root, ann_rel)

            # 检查图片是否存在，尝试 _test.jpg 后缀
            if not os.path.exists(img_full):
                base, ext = os.path.splitext(img_full)
                for suffix in ["_test.jpg", "_test.png", ".jpg", ".png"]:
                    alt = base + suffix
                    if os.path.exists(alt):
                        img_full = alt
                        break

            if os.path.exists(img_full) and os.path.exists(ann_full):
                pairs.append((img_full, ann_full))
    return pairs


def parse_annotations(annot_path):
    """解析标注文件，返回 bbox 和 label 列表"""
    boxes, labels = [], []
    with open(annot_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 5:
                continue
            x1, y1, x2, y2 = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            raw_type = int(parts[4])
            if raw_type not in CLASS_MAP:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(CLASS_MAP[raw_type])
    return boxes, labels


class PCBDataset(Dataset):
    """PCB缺陷检测数据集（含随机翻转增强）"""

    def __init__(self, image_annot_pairs, train=True):
        self.pairs = image_annot_pairs
        self.train = train

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, ann_path = self.pairs[idx]

        image = Image.open(img_path).convert("RGB")
        img_w, img_h = image.size
        boxes, labels = parse_annotations(ann_path)

        # 训练时随机水平翻转
        if self.train and random.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            for box in boxes:
                x1, y1, x2, y2 = box
                box[0], box[2] = img_w - x2, img_w - x1

        # 转为 tensor（保持绝对像素坐标，torchvision 内部自己处理）
        boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
        labels_t = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes_t,
            "labels": labels_t + 1,  # 0=背景，所以+1
            "image_id": torch.as_tensor([idx], dtype=torch.int64),
            "area": (boxes_t[:, 2] - boxes_t[:, 0]) * (boxes_t[:, 3] - boxes_t[:, 1])
                    if len(boxes) > 0 else torch.zeros(0),
            "iscrowd": torch.zeros(len(boxes), dtype=torch.int64),
        }

        # 亮度抖动（训练时）
        if self.train:
            image = T.ColorJitter(brightness=0.2, contrast=0.1)(image)
        image = T.ToTensor()(image)
        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))


def create_dataloaders(batch_size=4, num_workers=0):
    """创建检测任务的 DataLoader"""
    pcbdata_root = find_pcbdata_root()
    print(f"PCBData 根目录: {pcbdata_root}")

    trainval_path = os.path.join(pcbdata_root, "trainval.txt")
    test_path = os.path.join(pcbdata_root, "test.txt")

    # 解析全部配对
    all_trainval = parse_split_file(trainval_path, pcbdata_root)
    all_test = parse_split_file(test_path, pcbdata_root)

    # trainval 中随机取 80% 为 train，20% 为 val
    random.shuffle(all_trainval)
    n_train = int(len(all_trainval) * 0.8)
    train_pairs = all_trainval[:n_train]
    val_pairs = all_trainval[n_train:]

    print(f"训练集: {len(train_pairs)} 张  |  验证集: {len(val_pairs)} 张  |  测试集: {len(all_test)} 张")

    train_dataset = PCBDataset(train_pairs, train=True)
    val_dataset = PCBDataset(val_pairs, train=False)
    test_dataset = PCBDataset(all_test, train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = create_dataloaders()
    for images, targets in train_loader:
        print(f"Batch: {len(images)} images")
        print(f"  Image shape: {images[0].shape}")
        print(f"  Boxes: {targets[0]['boxes'].shape}")
        print(f"  Labels: {targets[0]['labels']}")
        break
