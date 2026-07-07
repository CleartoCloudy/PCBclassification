"""
DeepPCB数据集解析与缺陷裁剪模块
功能：读取原始DeepPCB标注 → 裁剪缺陷ROI → 按类别保存 → 划分train/val/test
Author: CleartoCloudy

支持两种目录结构：
  1. DeepPCB_original/group00041/00041/ (+ 00041_not/)  [GitHub克隆结构]
  2. DeepPCB_original/DeepPCB/PCBData/group00041/...    [含DeepPCB父目录]
"""
import os
import sys
import random
from tqdm import tqdm
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATASET_ORIGINAL, DATASET_PROCESSED, CLASS_NAMES,
                    TRAIN_RATIO, VAL_RATIO, TEST_RATIO, IMG_SIZE_TRAD, RANDOM_SEED)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# 标注中的类型号为 1~6，需减1映射到 CLASS_NAMES 的 0~5
# 1=open, 2=short, 3=mousebite, 4=spur, 5=pinhole, 6=spurious_copper


def find_data_root():
    """
    自动查找PCBData根目录，兼容不同解压层级
    返回: 包含 groupXXXXX 子目录的实际数据路径
    """
    # 候选路径（按优先级）
    candidates = [
        os.path.join(DATASET_ORIGINAL, "DeepPCB", "PCBData"),   # git clone 后有 DeepPCB 父目录
        DATASET_ORIGINAL,                                        # 数据直接放在此目录
    ]

    for cand in candidates:
        if os.path.isdir(cand):
            # 检查是否包含 group* 目录
            for name in os.listdir(cand):
                full = os.path.join(cand, name)
                if os.path.isdir(full) and name.startswith("group"):
                    return cand
    return None


def discover_groups(data_root):
    """
    扫描 data_root 下的 group* 目录，建立 标注→图像 的映射列表。

    目录结构示意:
      group00041/
        ├── 00041/          ← 图像目录（_test.jpg 或 .jpg）
        └── 00041_not/      ← 标注目录（.txt）

    返回: [(annot_path, images_dir), ...]
    """
    pairs = []
    for group_name in sorted(os.listdir(data_root)):
        group_dir = os.path.join(data_root, group_name)
        if not os.path.isdir(group_dir) or not group_name.startswith("group"):
            continue

        # 找到标注子目录 (*_not) 和图像子目录 (同名不加 _not)
        annot_dir = None
        images_dir = None
        for sub in os.listdir(group_dir):
            sub_path = os.path.join(group_dir, sub)
            if not os.path.isdir(sub_path):
                continue
            if sub.endswith("_not"):
                annot_dir = sub_path
            else:
                images_dir = sub_path

        if annot_dir and images_dir:
            for txt_file in sorted(os.listdir(annot_dir)):
                if txt_file.endswith(".txt"):
                    pairs.append((os.path.join(annot_dir, txt_file), images_dir))

    return pairs


def parse_annotation_line(line):
    """
    解析一行标注，支持空格或逗号分隔
    标注格式: x1 y1 x2 y2 type (type 范围 1~6)
    返回: (x1, y1, x2, y2, defect_type 0~5)，或 None
    """
    line = line.strip()
    if not line:
        return None
    sep = "," if "," in line else None
    parts = line.split(sep)
    if len(parts) < 5:
        return None
    try:
        x1, y1, x2, y2 = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
        dtype = int(parts[4]) - 1  # 1~6 → 0~5
    except ValueError:
        return None
    if not (0 <= dtype < len(CLASS_NAMES)):
        return None
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    return (x1, y1, x2, y2, dtype)


def parse_annotations(annot_path):
    """读取标注文件，返回缺陷列表"""
    defects = []
    with open(annot_path, "r") as f:
        for line in f:
            result = parse_annotation_line(line)
            if result:
                defects.append(result)
    return defects


def find_test_image(images_dir, base_name):
    """
    根据标注文件的 base_name 查找对应的缺陷图像。
    优先匹配 `<base>_test.jpg`，其次 `<base>.jpg`。
    """
    candidates = [
        f"{base_name}_test.jpg",
        f"{base_name}_test.png",
        f"{base_name}.jpg",
        f"{base_name}.png",
    ]
    for cand in candidates:
        cand_path = os.path.join(images_dir, cand)
        if os.path.exists(cand_path):
            return cand_path
    return None


def crop_defect_region(image, bbox, padding=5):
    """
    从图像中裁剪缺陷区域，带padding
    返回裁剪区域，或None（bbox无效时）
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    if x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2]


def resize_with_pad(image, target_size):
    """
    调整图像到目标尺寸，短边等比缩放后填充黑边
    """
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    if len(resized.shape) == 2:
        canvas = np.zeros((target_size, target_size), dtype=np.uint8)
    else:
        canvas = np.zeros((target_size, target_size, resized.shape[2]), dtype=np.uint8)
    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    return canvas


def collect_all_samples():
    """
    遍历所有标注文件，裁剪所有缺陷样本
    返回: {class_id: [image, ...], ...}
    """
    data_root = find_data_root()
    if data_root is None:
        print(f"[错误] 在 {DATASET_ORIGINAL} 中未找到DeepPCB数据集")
        print("请确保目录结构为:")
        print(f"  {DATASET_ORIGINAL}/DeepPCB/PCBData/group00041/...")
        print(f"  或 {DATASET_ORIGINAL}/group00041/...")
        sys.exit(1)

    print(f"数据根目录: {data_root}")
    pairs = discover_groups(data_root)
    if not pairs:
        print(f"[错误] 未找到任何标注/图像配对")
        sys.exit(1)

    print(f"找到 {len(pairs)} 个标注文件")

    samples = {i: [] for i in range(len(CLASS_NAMES))}
    skipped = 0

    for annot_path, images_dir in tqdm(pairs, desc="解析标注 & 裁剪缺陷"):
        defects = parse_annotations(annot_path)
        if not defects:
            continue

        base_name = os.path.splitext(os.path.basename(annot_path))[0]
        img_path = find_test_image(images_dir, base_name)
        if img_path is None:
            skipped += 1
            continue

        image = cv2.imread(img_path)
        if image is None:
            skipped += 1
            continue

        for x1, y1, x2, y2, class_id in defects:
            cropped = crop_defect_region(image, (x1, y1, x2, y2))
            if cropped is None:
                continue
            cropped = resize_with_pad(cropped, IMG_SIZE_TRAD)
            samples[class_id].append(cropped)

    if skipped > 0:
        print(f"警告: {skipped} 个标注文件找不到对应图像，已跳过")

    total = sum(len(v) for v in samples.values())
    print(f"\n共裁剪 {total} 个缺陷样本")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name}: {len(samples[i])} 张")
    return samples


def split_and_save(samples):
    """按train/val/test划分并保存到分类目录"""
    for split in ["train", "val", "test"]:
        for cls_name in CLASS_NAMES:
            os.makedirs(os.path.join(DATASET_PROCESSED, split, cls_name), exist_ok=True)

    for cls_id, cls_name in enumerate(CLASS_NAMES):
        imgs = samples[cls_id]
        random.shuffle(imgs)
        n = len(imgs)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)

        train_imgs = imgs[:n_train]
        val_imgs = imgs[n_train:n_train + n_val]
        test_imgs = imgs[n_train + n_val:]

        for split, split_imgs in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
            for idx, img in enumerate(split_imgs):
                save_path = os.path.join(DATASET_PROCESSED, split, cls_name,
                                         f"{cls_name}_{idx:05d}.jpg")
                cv2.imwrite(save_path, img)

        print(f"  {cls_name}: train={len(train_imgs)}, val={len(val_imgs)}, test={len(test_imgs)}")

    print(f"\n数据集已保存至: {DATASET_PROCESSED}")


def main():
    print("=" * 60)
    print("DeepPCB 缺陷裁剪与数据集划分")
    print("=" * 60)
    samples = collect_all_samples()
    split_and_save(samples)
    print("完成！")


if __name__ == "__main__":
    main()
