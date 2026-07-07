"""
统一评估模块 - 对所有模型进行测试集评估并汇总对比
Author: CleartoCloudy
"""
import os
import sys
import json
import time
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATASET_PROCESSED, CLASS_NAMES, LOGS_DIR, MODELS_DIR, DEVICE, BATCH_SIZE, RANDOM_SEED

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

from deep_learning.dataset import PCBDataset, get_val_transforms, create_test_loader
from deep_learning.train import build_model
from deep_learning.models.resnet18 import count_parameters as count_resnet_params
from deep_learning.models.mobilenetv3 import count_parameters as count_mobilenet_params
from traditional.hog_svm import test_hog_svm
from traditional.lbp_svm import test_lbp_svm


def evaluate_all_models():
    """
    对所有模型进行评估并汇总结果
    返回汇总字典供可视化使用
    """
    num_workers = 0 if os.name == 'nt' else 4

    # 确定测试集目录
    test_dir = os.path.join(DATASET_PROCESSED, "test_clahe")
    if not os.path.exists(test_dir):
        test_dir = os.path.join(DATASET_PROCESSED, "test")

    print("=" * 70)
    print("统一模型评估")
    print("=" * 70)

    all_results = {}

    # ---- 实验一: HOG + SVM ----
    print("\n>>> 实验一: HOG + SVM")
    try:
        hog_metrics = test_hog_svm(test_dir)
        all_results["HOG+SVM"] = {
            "accuracy": hog_metrics["accuracy"],
            "precision": hog_metrics["precision"],
            "recall": hog_metrics["recall"],
            "f1": hog_metrics["f1"],
            "confusion_matrix": hog_metrics["confusion_matrix"],
            "y_true": hog_metrics["y_true"],
            "y_pred": hog_metrics["y_pred"],
            "params_M": 0,
            "inference_time_ms": 0,
            "type": "traditional",
        }
        print(f"  Acc: {hog_metrics['accuracy']:.4f}, F1: {hog_metrics['f1']:.4f}")
    except Exception as e:
        print(f"  [跳过] {e}")

    # ---- 实验二: LBP + SVM ----
    print("\n>>> 实验二: LBP + SVM")
    try:
        lbp_metrics = test_lbp_svm(test_dir)
        all_results["LBP+SVM"] = {
            "accuracy": lbp_metrics["accuracy"],
            "precision": lbp_metrics["precision"],
            "recall": lbp_metrics["recall"],
            "f1": lbp_metrics["f1"],
            "confusion_matrix": lbp_metrics["confusion_matrix"],
            "y_true": lbp_metrics["y_true"],
            "y_pred": lbp_metrics["y_pred"],
            "params_M": 0,
            "inference_time_ms": 0,
            "type": "traditional",
        }
        print(f"  Acc: {lbp_metrics['accuracy']:.4f}, F1: {lbp_metrics['f1']:.4f}")
    except Exception as e:
        print(f"  [跳过] {e}")

    # ---- 深度学习模型 ----
    dl_models = {
        "ResNet18": "resnet18",
        "MobileNetV3": "mobilenetv3",
        "ResNet18+SE": "resnet18_se",
        "ResNet18+CBAM": "resnet18_cbam",
    }

    test_loader, n_test = create_test_loader(test_dir, BATCH_SIZE, num_workers)

    for display_name, model_type in dl_models.items():
        model_key = None
        # 区分实验编号
        if model_type == "resnet18":
            exp = "实验三"
        elif model_type == "mobilenetv3":
            exp = "实验四"
        else:
            exp = "实验五"

        print(f"\n>>> {exp}: {display_name}")

        try:
            model = build_model(model_type)
            model_path = os.path.join(MODELS_DIR, f"{model_type}_best.pth")

            if not os.path.exists(model_path):
                print(f"  [跳过] 模型文件不存在: {model_path}")
                continue

            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            model = model.to(DEVICE)
            model.eval()

            # 统计参数量
            if "resnet" in model_type:
                params_M = count_resnet_params(model)
            else:
                params_M = count_mobilenet_params(model)

            # 推理时间评估
            all_preds, all_labels = [], []
            inference_times = []

            for images, labels in tqdm(test_loader, desc=f"测试 {display_name}"):
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                t0 = time.time()
                with torch.no_grad():
                    outputs = model(images)
                inference_times.append(time.time() - t0)

                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            avg_inference = np.mean(inference_times) * 1000

            all_results[display_name] = {
                "accuracy": accuracy_score(all_labels, all_preds),
                "precision": precision_score(all_labels, all_preds, average="macro", zero_division=0),
                "recall": recall_score(all_labels, all_preds, average="macro", zero_division=0),
                "f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
                "confusion_matrix": confusion_matrix(all_labels, all_preds),
                "y_true": all_labels,
                "y_pred": all_preds,
                "params_M": round(params_M, 2),
                "inference_time_ms": round(avg_inference, 2),
                "type": "deep_learning",
            }
            print(f"  Acc: {all_results[display_name]['accuracy']:.4f}, "
                  f"F1: {all_results[display_name]['f1']:.4f}, "
                  f"Params: {params_M:.2f}M")

        except Exception as e:
            print(f"  [跳过] {e}")

    # 保存汇总结果
    summary = {}
    for name, metrics in all_results.items():
        summary[name] = {
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "params_M": metrics["params_M"],
            "inference_time_ms": metrics["inference_time_ms"],
            "type": metrics["type"],
            "confusion_matrix": metrics["confusion_matrix"].tolist() if hasattr(metrics["confusion_matrix"], "tolist") else metrics["confusion_matrix"],
        }

    summary_path = os.path.join(LOGS_DIR, "all_results_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n汇总结果已保存至: {summary_path}")

    return all_results


def main():
    evaluate_all_models()


if __name__ == "__main__":
    main()
