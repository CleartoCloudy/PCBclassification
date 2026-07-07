"""
数据增强模块 - 对训练集应用旋转、翻转、亮度变化、高斯噪声
Author: CleartoCloudy
"""
import os
import sys
import random
import cv2
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATASET_PROCESSED, CLASS_NAMES,
                    AUG_ROTATION_DEG, AUG_BRIGHTNESS_RANGE,
                    AUG_NOISE_STD, AUG_PER_SAMPLE, RANDOM_SEED)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def random_rotation(image, max_deg=15):
    """随机旋转"""
    angle = random.uniform(-max_deg, max_deg)
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)


def random_flip(image):
    """随机水平翻转（PCB缺陷方向无关）"""
    return cv2.flip(image, 1) if random.random() > 0.5 else image


def random_brightness(image, range_val=0.2):
    """随机亮度调整"""
    delta = random.uniform(-range_val, range_val)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (1 + delta), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def random_noise(image, std=0.02):
    """添加高斯噪声"""
    noise = np.random.normal(0, std * 255, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def augment_image(image):
    """
    对单张图像应用随机增强组合
    返回增强后的图像
    """
    aug = image.copy()
    if random.random() > 0.5:
        aug = random_rotation(aug, AUG_ROTATION_DEG)
    if random.random() > 0.5:
        aug = random_flip(aug)
    if random.random() > 0.5:
        aug = random_brightness(aug, AUG_BRIGHTNESS_RANGE)
    if random.random() > 0.5:
        aug = random_noise(aug, AUG_NOISE_STD)
    return aug


def augment_training_set(target_size=None):
    """
    对训练集（包括CLAHE增强后的）进行数据增强
    若无CLAHE目录则使用原始训练集
    """
    # 优先使用CLAHE增强后的数据
    train_clahe = os.path.join(DATASET_PROCESSED, "train_clahe")
    train_orig = os.path.join(DATASET_PROCESSED, "train")
    input_dir = train_clahe if os.path.exists(train_clahe) else train_orig
    output_dir = os.path.join(DATASET_PROCESSED, "train_augmented")

    if not os.path.exists(input_dir):
        print(f"训练集目录不存在: {input_dir}")
        return

    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"每张图片生成 {AUG_PER_SAMPLE} 个增强样本")

    for cls_name in CLASS_NAMES:
        os.makedirs(os.path.join(output_dir, cls_name), exist_ok=True)

    for cls_name in CLASS_NAMES:
        class_dir = os.path.join(input_dir, cls_name)
        if not os.path.exists(class_dir):
            continue
        images = [f for f in os.listdir(class_dir)
                  if f.endswith((".jpg", ".png", ".bmp"))]

        for img_name in tqdm(images, desc=f"增强 {cls_name}"):
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            # 保留原始图
            out_name = os.path.splitext(img_name)[0]
            cv2.imwrite(os.path.join(output_dir, cls_name, f"{out_name}_orig.jpg"), img)

            # 生成增强样本
            for aug_idx in range(AUG_PER_SAMPLE):
                aug_img = augment_image(img)
                cv2.imwrite(
                    os.path.join(output_dir, cls_name, f"{out_name}_aug{aug_idx}.jpg"),
                    aug_img
                )

    # 统计
    for cls_name in CLASS_NAMES:
        class_dir = os.path.join(output_dir, cls_name)
        if os.path.exists(class_dir):
            count = len([f for f in os.listdir(class_dir) if f.endswith(".jpg")])
            print(f"  {cls_name}: {count} 张")


def main():
    print("=" * 60)
    print("数据增强")
    print("=" * 60)
    augment_training_set()
    print("\n数据增强完成！")


if __name__ == "__main__":
    main()
