"""
PCB缺陷分类系统 - 全局配置文件
Author: CleartoCloudy
"""
import os

# ==================== 路径配置 ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_ORIGINAL = os.path.join(BASE_DIR, "datasets", "DeepPCB_original")
DATASET_PROCESSED = os.path.join(BASE_DIR, "datasets", "PCB_Classification")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
CM_DIR = os.path.join(RESULTS_DIR, "confusion_matrix")
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")
MODELS_DIR = os.path.join(RESULTS_DIR, "models")

for d in [DATASET_PROCESSED, FIGURES_DIR, CM_DIR, LOGS_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

# ==================== 数据集配置 ====================
CLASS_NAMES = ["open", "short", "mousebite", "spur", "pinhole", "spurious_copper"]
NUM_CLASSES = 6

# 数据集划分比例
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# 图像尺寸
IMG_SIZE_TRAD = 128   # 传统方法输入尺寸
IMG_SIZE_DL = 224     # 深度学习方法输入尺寸

# ==================== 数据增强配置 ====================
AUGMENTATION_ENABLED = True
AUG_ROTATION_DEG = 15        # 旋转角度范围
AUG_BRIGHTNESS_RANGE = 0.2   # 亮度变化范围
AUG_NOISE_STD = 0.02         # 高斯噪声标准差
AUG_PER_SAMPLE = 4           # 每张原始图生成的增强样本数

# ==================== CLAHE配置 ====================
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

# ==================== HOG配置 ====================
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)

# ==================== LBP配置 ====================
LBP_RADIUS = 3
LBP_N_POINTS = 24
LBP_METHOD = "uniform"

# ==================== SVM配置 ====================
SVM_C = 1.0
SVM_KERNEL = "linear"
SVM_GAMMA = "scale"
SVM_MAX_ITER = 5000

# ==================== 深度学习训练配置 ====================
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
LR_STEP_SIZE = 15
LR_GAMMA = 0.1

# 早停配置
EARLY_STOPPING_PATIENCE = 10

# 设备配置
try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    torch = None
    DEVICE = "cpu"

# ==================== 随机种子 ====================
RANDOM_SEED = 42
