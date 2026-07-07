"""
CLAHE（自适应直方图均衡化）图像增强模块
对所有分类数据集图像应用CLAHE增强，提升局部对比度
Author: CleartoCloudy
"""
import os
import sys
import cv2
import numpy as np
from tqdm import tqdm
from multiprocessing import cpu_count

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATASET_PROCESSED, CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID_SIZE, CLASS_NAMES


def apply_clahe(image):
    """对BGR图像应用CLAHE（仅在亮度通道）"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT,
                            tileGridSize=CLAHE_TILE_GRID_SIZE)
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge((l_eq, a, b))
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def clahe_enhance_directory(input_dir, output_dir):
    """对目录中所有图像应用CLAHE"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    for cls_name in CLASS_NAMES:
        os.makedirs(os.path.join(output_dir, cls_name), exist_ok=True)

    for cls_name in CLASS_NAMES:
        class_input = os.path.join(input_dir, cls_name)
        class_output = os.path.join(output_dir, cls_name)
        if not os.path.exists(class_input):
            continue
        images = [f for f in os.listdir(class_input)
                  if f.endswith((".jpg", ".png", ".bmp"))]
        for img_name in tqdm(images, desc=f"CLAHE {cls_name}"):
            img_path = os.path.join(class_input, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            enhanced = apply_clahe(img)
            cv2.imwrite(os.path.join(class_output, img_name), enhanced)


def main():
    print("=" * 60)
    print("CLAHE 图像增强")
    print("=" * 60)

    for split in ["train", "val", "test"]:
        input_dir = os.path.join(DATASET_PROCESSED, split)
        output_dir = os.path.join(DATASET_PROCESSED, f"{split}_clahe")
        if not os.path.exists(input_dir):
            print(f"跳过 {split}: 目录不存在")
            continue
        print(f"\n处理 {split} 集...")
        clahe_enhance_directory(input_dir, output_dir)

    print("\nCLAHE增强完成！")


if __name__ == "__main__":
    main()
