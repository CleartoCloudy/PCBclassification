"""
传统方法通用特征提取工具
Author: CleartoCloudy
"""
import os
import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (HOG_ORIENTATIONS, HOG_PIXELS_PER_CELL, HOG_CELLS_PER_BLOCK,
                    LBP_RADIUS, LBP_N_POINTS, LBP_METHOD, CLASS_NAMES)


def extract_hog_features(image, target_size=(128, 128)):
    """
    提取HOG特征
    image: BGR图像 (H, W, 3)
    返回: 特征向量
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, target_size)
    features = hog(gray,
                   orientations=HOG_ORIENTATIONS,
                   pixels_per_cell=HOG_PIXELS_PER_CELL,
                   cells_per_block=HOG_CELLS_PER_BLOCK,
                   block_norm='L2-Hys',
                   visualize=False)
    return features


def extract_lbp_features(image, target_size=(128, 128)):
    """
    提取LBP特征（直方图）
    image: BGR图像
    返回: LBP直方图特征向量
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, target_size)
    lbp = local_binary_pattern(gray, LBP_N_POINTS, LBP_RADIUS, method=LBP_METHOD)
    n_bins = LBP_N_POINTS + 2 if LBP_METHOD == "uniform" else int(lbp.max() + 1)

    # 分块统计直方图，获取空间信息
    h, w = lbp.shape
    hist = []
    grid_size = 4
    for i in range(grid_size):
        for j in range(grid_size):
            block = lbp[i * h // grid_size:(i + 1) * h // grid_size,
                         j * w // grid_size:(j + 1) * w // grid_size]
            block_hist, _ = np.histogram(block, bins=n_bins, range=(0, n_bins))
            hist.append(block_hist.astype(np.float32))

    hist = np.concatenate(hist)
    hist /= (hist.sum() + 1e-7)
    return hist


def load_dataset_from_directory(directory, feature_extractor, target_size=(128, 128)):
    """
    从分类目录加载数据并提取特征
    directory: 包含类别子文件夹的目录
    feature_extractor: 特征提取函数，签名为 (image, target_size) -> feature_vector
    返回: X (特征矩阵), y (标签), paths (图像路径列表)
    """
    X, y, paths = [], [], []
    for class_id, cls_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(directory, cls_name)
        if not os.path.exists(class_dir):
            continue
        images = [f for f in os.listdir(class_dir) if f.endswith((".jpg", ".png", ".bmp"))]
        for img_name in tqdm(images, desc=f"提取 {cls_name} 特征"):
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            features = feature_extractor(img, target_size)
            X.append(features)
            y.append(class_id)
            paths.append(img_path)

    return np.array(X), np.array(y), paths
