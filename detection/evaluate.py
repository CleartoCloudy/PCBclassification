"""
检测模型评估 — mAP@0.5, 各类别 AP
Author: CleartoCloudy
"""
import os
import sys
import json
import time
import numpy as np
import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODELS_DIR, LOGS_DIR, CLASS_NAMES, DEVICE, BASE_DIR
from detection.dataset import create_dataloaders
from detection.train import evaluate as evaluate_mAP
from detection.model import create_model, count_parameters


def evaluate_detection_model():
    """评估检测模型在测试集上的性能"""
    print("\n" + "=" * 60)
    print("PCB缺陷检测 — 模型评估")
    print("=" * 60)

    # 加载模型
    model_path = os.path.join(MODELS_DIR, "detection_faster_rcnn_best.pth")
    if not os.path.exists(model_path):
        print(f"[错误] 模型文件不存在: {model_path}")
        print("请先运行: python detection/train.py")
        return

    model = create_model(num_classes=len(CLASS_NAMES) + 1, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    params_M = count_parameters(model)
    print(f"模型: Faster R-CNN (ResNet50-FPN)  |  参数量: {params_M:.2f}M")

    # 加载数据
    num_workers = 0 if os.name == "nt" else 4
    _, _, test_loader = create_dataloaders(batch_size=1, num_workers=num_workers)

    # 评估
    print("\n评估中...")
    t0 = time.time()
    mAP = evaluate_mAP(model, test_loader, DEVICE)
    eval_time = time.time() - t0

    print(f"\n测试集 mAP@0.5: {mAP:.4f}")
    print(f"评估耗时: {eval_time:.1f}s")

    # 保存
    result = {
        "model": "Faster R-CNN (ResNet50-FPN)",
        "mAP50": round(mAP, 4),
        "params_M": round(params_M, 2),
        "eval_time_s": round(eval_time, 1),
    }
    with open(os.path.join(LOGS_DIR, "detection_result.json"), "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存至: {LOGS_DIR}/detection_result.json")

    return result


if __name__ == "__main__":
    evaluate_detection_model()
