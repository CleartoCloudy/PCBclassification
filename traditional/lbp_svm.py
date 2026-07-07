"""
LBP + SVM 纹理特征分类方法
Author: CleartoCloudy
"""
import os
import sys
import pickle
import time
from copy import deepcopy
import numpy as np
from tqdm import tqdm
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATASET_PROCESSED, MODELS_DIR, LOGS_DIR,
                    SVM_C, SVM_KERNEL, SVM_GAMMA, SVM_MAX_ITER, IMG_SIZE_TRAD, RANDOM_SEED)
from traditional.features import extract_lbp_features, load_dataset_from_directory

np.random.seed(RANDOM_SEED)


def train_lbp_svm(train_dir, val_dir=None):
    """
    训练LBP+SVM模型
    """
    print("\n" + "=" * 60)
    print("实验二: LBP + SVM")
    print("=" * 60)

    # 1. 提取LBP特征
    print("\n[1/3] 提取LBP特征...")
    t0 = time.time()
    X_train, y_train, _ = load_dataset_from_directory(train_dir, extract_lbp_features, (IMG_SIZE_TRAD, IMG_SIZE_TRAD))
    feature_time = time.time() - t0
    print(f"  训练集特征维度: {X_train.shape}")
    print(f"  特征提取耗时: {feature_time:.2f}s")

    # 2. 归一化 & 训练SVM
    print("\n[2/3] 训练SVM分类器...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # 提前准备验证集用于训练中监控
    X_val_scaled, y_val = None, None
    if val_dir and os.path.exists(val_dir):
        print("  加载验证集用于训练监控...")
        X_val, y_val, _ = load_dataset_from_directory(
            val_dir, extract_lbp_features, (IMG_SIZE_TRAD, IMG_SIZE_TRAD))
        X_val_scaled = scaler.transform(X_val)

    t0 = time.time()
    if SVM_KERNEL == "linear":
        print(f"  使用 SGDClassifier (线性SVM) + 进度监控")
        svm = SGDClassifier(
            loss="hinge", alpha=1.0 / (SVM_C * len(X_train_scaled)),
            max_iter=1, tol=None, warm_start=True,
            random_state=RANDOM_SEED)
        svm.fit(X_train_scaled, y_train)  # 初始化 classes_

        best_val_acc = 0.0
        best_svm = None
        n_no_improve = 0
        pbar = tqdm(range(1, SVM_MAX_ITER + 1), desc="  SVM 训练",
                     bar_format="{l_bar}{bar:30}{r_bar}", ncols=110)

        for epoch in pbar:
            svm.partial_fit(X_train_scaled, y_train)
            train_pred = svm.predict(X_train_scaled)
            train_acc = accuracy_score(y_train, train_pred)

            if X_val_scaled is not None:
                val_pred = svm.predict(X_val_scaled)
                val_acc = accuracy_score(y_val, val_pred)
                pbar.set_postfix_str(
                    f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f}")

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_svm = deepcopy(svm)
                    n_no_improve = 0
                else:
                    n_no_improve += 1
                    if n_no_improve >= 15:
                        pbar.set_postfix_str(
                            f"train_acc={train_acc:.4f}  val_acc={best_val_acc:.4f}  收敛")
                        break
            else:
                pbar.set_postfix_str(f"train_acc={train_acc:.4f}")

        if best_svm is not None:
            svm = best_svm
        train_time = time.time() - t0
        actual_epochs = epoch
    else:
        print(f"  使用 SVC (kernel={SVM_KERNEL}) — 正在训练, 请稍候...")
        svm = SVC(C=SVM_C, kernel=SVM_KERNEL, gamma=SVM_GAMMA,
                  probability=True, random_state=RANDOM_SEED, cache_size=1000)
        svm.fit(X_train_scaled, y_train)
        train_time = time.time() - t0
        actual_epochs = "—"

    print(f"  训练完成 | 耗时: {train_time:.1f}s | 轮次: {actual_epochs}")

    y_train_pred = svm.predict(X_train_scaled)
    train_metrics = {
        "accuracy": accuracy_score(y_train, y_train_pred),
        "precision": precision_score(y_train, y_train_pred, average="macro", zero_division=0),
        "recall": recall_score(y_train, y_train_pred, average="macro", zero_division=0),
        "f1": f1_score(y_train, y_train_pred, average="macro", zero_division=0),
        "time": train_time,
    }
    print(f"  训练完成, 耗时: {train_time:.2f}s")
    print(f"  训练集 Acc: {train_metrics['accuracy']:.4f}")

    # 3. 验证集评估
    val_metrics = None
    if val_dir and os.path.exists(val_dir):
        print("\n[3/3] 验证集评估...")
        X_val, y_val, _ = load_dataset_from_directory(val_dir, extract_lbp_features, (IMG_SIZE_TRAD, IMG_SIZE_TRAD))
        X_val_scaled = scaler.transform(X_val)
        y_val_pred = svm.predict(X_val_scaled)
        val_metrics = {
            "accuracy": accuracy_score(y_val, y_val_pred),
            "precision": precision_score(y_val, y_val_pred, average="macro", zero_division=0),
            "recall": recall_score(y_val, y_val_pred, average="macro", zero_division=0),
            "f1": f1_score(y_val, y_val_pred, average="macro", zero_division=0),
            "confusion_matrix": confusion_matrix(y_val, y_val_pred),
            "y_true": y_val,
            "y_pred": y_val_pred,
        }
        print(f"  验证集 Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}")

    # 4. 保存模型
    model_path = os.path.join(MODELS_DIR, "lbp_svm.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({"svm": svm, "scaler": scaler}, f)
    print(f"  模型已保存至: {model_path}")

    # 保存训练信息供报告使用
    import json
    hist_path = os.path.join(LOGS_DIR, "lbp_svm_history.json")
    with open(hist_path, "w") as f:
        json.dump({"train_time_s": round(feature_time + train_time, 1)}, f)

    return svm, scaler, train_metrics, val_metrics


def test_lbp_svm(test_dir, model_path=None):
    """测试LBP+SVM模型"""
    if model_path is None:
        model_path = os.path.join(MODELS_DIR, "lbp_svm.pkl")

    with open(model_path, "rb") as f:
        data = pickle.load(f)

    svm = data["svm"]
    scaler = data["scaler"]

    X_test, y_test, _ = load_dataset_from_directory(test_dir, extract_lbp_features, (IMG_SIZE_TRAD, IMG_SIZE_TRAD))
    X_test_scaled = scaler.transform(X_test)
    y_pred = svm.predict(X_test_scaled)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "y_true": y_test,
        "y_pred": y_pred,
    }
    return metrics


def main():
    train_dir = os.path.join(DATASET_PROCESSED, "train_augmented")
    if not os.path.exists(train_dir):
        train_dir = os.path.join(DATASET_PROCESSED, "train_clahe")
    if not os.path.exists(train_dir):
        train_dir = os.path.join(DATASET_PROCESSED, "train")

    val_dir = os.path.join(DATASET_PROCESSED, "val_clahe")
    if not os.path.exists(val_dir):
        val_dir = os.path.join(DATASET_PROCESSED, "val")

    train_lbp_svm(train_dir, val_dir)


if __name__ == "__main__":
    main()
