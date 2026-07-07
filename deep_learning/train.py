"""
深度学习模型统一训练脚本
支持: ResNet18, MobileNetV3, ResNet18+SE, ResNet18+CBAM
Author: CleartoCloudy
"""
import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATASET_PROCESSED, MODELS_DIR, LOGS_DIR,
                    BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
                    LR_STEP_SIZE, LR_GAMMA, EARLY_STOPPING_PATIENCE,
                    DEVICE, NUM_CLASSES, RANDOM_SEED, CLASS_NAMES)
from deep_learning.dataset import create_dataloaders, create_test_loader
from deep_learning.models.resnet18 import create_resnet18
from deep_learning.models.mobilenetv3 import create_mobilenetv3
from deep_learning.models.resnet18 import count_parameters as count_resnet_params
from deep_learning.models.mobilenetv3 import count_parameters as count_mobilenet_params


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=10, mode="max"):
        self.patience = patience
        self.mode = mode
        self.best_score = -float("inf") if mode == "max" else float("inf")
        self.counter = 0
        self.best_state = None

    def __call__(self, score, model):
        improved = (self.mode == "max" and score > self.best_score) or \
                   (self.mode == "min" and score < self.best_score)
        if improved:
            self.best_score = score
            self.counter = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return False  # 未触发早停
        self.counter += 1
        return self.counter >= self.patience


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, total_epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    total_batches = len(loader)

    pbar = tqdm(enumerate(loader), total=total_batches, desc=f"Epoch {epoch}/{total_epochs}",
                bar_format="{l_bar}{bar:30}{r_bar}", ncols=120)
    for batch_idx, (images, labels) in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        current_loss = running_loss / total
        current_acc = correct / total
        pbar.set_postfix_str(f"loss={current_loss:.4f} acc={current_acc:.4f}")

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc, all_labels, all_preds


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc="  Val", bar_format="{l_bar}{bar:20}{r_bar}", ncols=100)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc, all_labels, all_preds


def train_model(model, train_loader, val_loader, model_name, num_epochs=NUM_EPOCHS):
    """统一训练流程"""
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, mode="max")

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    print(f"\n{'='*70}")
    print(f"  模型: {model_name}  |  设备: {DEVICE}  |  Epochs: {num_epochs}")
    print(f"  批次大小: {BATCH_SIZE}  |  学习率: {LEARNING_RATE}  |  早停: {EARLY_STOPPING_PATIENCE}")
    print(f"{'='*70}")

    t0 = time.time()
    for epoch in range(1, num_epochs + 1):
        lr_now = optimizer.param_groups[0]["lr"]
        train_loss, train_acc, _, _ = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE, epoch, num_epochs)
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, DEVICE)

        scheduler.step()

        history["train_loss"].append(round(train_loss, 4))
        history["train_acc"].append(round(train_acc, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_acc"].append(round(val_acc, 4))

        # 清晰的单行汇总
        print(f"  Epoch {epoch:3d}/{num_epochs} │ "
              f"loss: {train_loss:.4f}→{val_loss:.4f} │ "
              f"acc: {train_acc:.4f}→{val_acc:.4f} │ "
              f"lr: {lr_now:.2e} {'│ ★ best' if val_acc >= best_val_acc else ''}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"{model_name}_best.pth"))

        if early_stopping(val_acc, model):
            print(f"  早停触发于 Epoch {epoch+1}")
            break

    train_time = time.time() - t0
    print(f"训练完成, 总耗时: {train_time:.2f}s, 最佳Val Acc: {best_val_acc:.4f}")

    # 加载最佳模型
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, f"{model_name}_best.pth")))

    # 保存训练历史
    history_path = os.path.join(LOGS_DIR, f"{model_name}_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    return model, history, train_time


@torch.no_grad()
def evaluate_model(model, test_loader, device=DEVICE):
    """评估模型性能，返回完整指标"""
    model.eval()
    all_preds, all_labels = [], []
    inference_times = []

    for images, labels in tqdm(test_loader, desc="测试评估"):
        images, labels = images.to(device), labels.to(device)

        t0 = time.time()
        outputs = model(images)
        inference_times.append(time.time() - t0)

        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_inference_time = np.mean(inference_times) * 1000  # 转为ms/batch

    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, average="macro", zero_division=0),
        "recall": recall_score(all_labels, all_preds, average="macro", zero_division=0),
        "f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(all_labels, all_preds),
        "y_true": all_labels,
        "y_pred": all_preds,
        "inference_time_ms": round(avg_inference_time, 2),
    }
    return metrics


def build_model(model_type):
    """根据类型构建模型"""
    model_type = model_type.lower()
    if model_type == "resnet18":
        return create_resnet18(num_classes=NUM_CLASSES, attention=None)
    elif model_type == "resnet18_se":
        return create_resnet18(num_classes=NUM_CLASSES, attention="se")
    elif model_type == "resnet18_cbam":
        return create_resnet18(num_classes=NUM_CLASSES, attention="cbam")
    elif model_type == "mobilenetv3":
        return create_mobilenetv3(num_classes=NUM_CLASSES)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")


def get_param_count(model_type):
    """获取模型参数量"""
    model_type = model_type.lower()
    if model_type.startswith("resnet"):
        model = create_resnet18(num_classes=NUM_CLASSES, attention=(
            "se" if "se" in model_type else ("cbam" if "cbam" in model_type else None)))
        return count_resnet_params(model)
    else:
        model = create_mobilenetv3(num_classes=NUM_CLASSES)
        return count_mobilenet_params(model)


def run_dl_experiment(model_type, train_dir=None, val_dir=None, test_dir=None):
    """运行一次深度学习实验（训练+验证+测试）"""
    set_seed(RANDOM_SEED)

    # 确定数据目录
    if train_dir is None:
        # 优先使用CLAHE增强数据
        train_dir = os.path.join(DATASET_PROCESSED, "train_augmented")
        if not os.path.exists(train_dir):
            train_dir = os.path.join(DATASET_PROCESSED, "train_clahe")
        if not os.path.exists(train_dir):
            train_dir = os.path.join(DATASET_PROCESSED, "train")
    if val_dir is None:
        val_dir = os.path.join(DATASET_PROCESSED, "val_clahe")
        if not os.path.exists(val_dir):
            val_dir = os.path.join(DATASET_PROCESSED, "val")
    if test_dir is None:
        test_dir = os.path.join(DATASET_PROCESSED, "test_clahe")
        if not os.path.exists(test_dir):
            test_dir = os.path.join(DATASET_PROCESSED, "test")

    # DataLoader设置 (Windows下num_workers=0避免多进程问题)
    num_workers = 0 if os.name == 'nt' else 4

    train_loader, val_loader, n_train, n_val = create_dataloaders(
        train_dir, val_dir, BATCH_SIZE, num_workers)
    test_loader, n_test = create_test_loader(test_dir, BATCH_SIZE, num_workers)

    # 构建模型
    model = build_model(model_type)
    params_M = get_param_count(model_type)
    print(f"模型: {model_type}  |  参数量: {params_M:.2f}M")

    # 训练
    model, history, train_time = train_model(model, train_loader, val_loader, model_type)

    # 测试
    print(f"\n测试集评估 ({model_type})...")
    test_metrics = evaluate_model(model, test_loader)

    # 汇总
    result = {
        "model": model_type,
        "params_M": round(params_M, 2),
        "train_time_s": round(train_time, 1),
        "train_acc": round(history["train_acc"][-1], 4),
        "val_acc": round(history["val_acc"][-1], 4),
        "test_acc": round(test_metrics["accuracy"], 4),
        "test_precision": round(test_metrics["precision"], 4),
        "test_recall": round(test_metrics["recall"], 4),
        "test_f1": round(test_metrics["f1"], 4),
        "inference_time_ms": test_metrics["inference_time_ms"],
        "confusion_matrix": test_metrics["confusion_matrix"].tolist(),
        "y_true": test_metrics["y_true"],
        "y_pred": test_metrics["y_pred"],
        "history": history,
    }

    # 保存结果
    result_path = os.path.join(LOGS_DIR, f"{model_type}_result.json")
    serializable = {k: v for k, v in result.items() if k not in ["y_true", "y_pred"]}
    with open(result_path, "w") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)

    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="深度学习模型训练")
    parser.add_argument("--model", type=str, required=True,
                        choices=["resnet18", "resnet18_se", "resnet18_cbam", "mobilenetv3"],
                        help="模型类型")
    args = parser.parse_args()
    run_dl_experiment(args.model)


if __name__ == "__main__":
    main()
