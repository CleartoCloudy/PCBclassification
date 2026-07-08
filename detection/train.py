"""
Faster R-CNN 检测模型训练
Author: CleartoCloudy
"""
import os
import sys
import math
import time
import json
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (MODELS_DIR, LOGS_DIR, DEVICE, RANDOM_SEED, CLASS_NAMES)
from detection.dataset import create_dataloaders
from detection.model import create_model, count_parameters

# 检测超参数
DET_BATCH_SIZE = 8
DET_NUM_EPOCHS = 30
DET_LR = 0.005
DET_LR_STEP = 8
DET_LR_GAMMA = 0.1
DET_EARLY_STOP = 10

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    def f(x):
        if x >= warmup_iters:
            return 1
        return warmup_factor + (1 - warmup_factor) * x / warmup_iters
    return optim.lr_scheduler.LambdaLR(optimizer, f)


@torch.no_grad()
def evaluate(model, data_loader, device):
    """在验证集上计算 mAP@0.5"""
    model.eval()
    all_preds, all_gts = [], []
    total_gt = 0
    total_pd = 0

    for images, targets in tqdm(data_loader, desc="  验证", leave=False):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(images)

        for output, target in zip(outputs, targets):
            total_gt += len(target["boxes"])
            # 统计 score > 0.05 的预测
            total_pd += (output["scores"] > 0.05).sum().item()
            all_preds.append({
                "boxes": output["boxes"].cpu(),
                "scores": output["scores"].cpu(),
                "labels": output["labels"].cpu(),
            })
            all_gts.append({
                "boxes": target["boxes"].cpu(),
                "labels": target["labels"].cpu(),
            })

    # 简化 mAP@0.5 计算（逐类平均）
    iou_thresh = 0.5
    aps = []
    for cls_id in range(1, len(CLASS_NAMES) + 1):
        # 收集该类别的所有预测和真值
        tp, fp, num_gt = [], [], 0
        for pred, gt in zip(all_preds, all_gts):
            gt_mask = gt["labels"] == cls_id
            num_gt += gt_mask.sum().item()
            pd_mask = pred["labels"] == cls_id

            if gt_mask.sum() == 0 and pd_mask.sum() == 0:
                continue

            gt_boxes = gt["boxes"][gt_mask]
            pd_boxes = pred["boxes"][pd_mask]
            pd_scores = pred["scores"][pd_mask]

            # 按置信度排序
            sorted_idx = torch.argsort(pd_scores, descending=True)
            pd_boxes = pd_boxes[sorted_idx]
            matched = torch.zeros(len(gt_boxes), dtype=torch.bool)

            for pb in pd_boxes:
                if len(gt_boxes) == 0:
                    fp.append(1)
                    tp.append(0)
                    continue
                ious = box_iou(pb.unsqueeze(0), gt_boxes)[0]
                best_iou, best_idx = ious.max(0)
                if best_iou >= iou_thresh and not matched[best_idx]:
                    tp.append(1)
                    fp.append(0)
                    matched[best_idx] = True
                else:
                    tp.append(0)
                    fp.append(1)

        if num_gt == 0 and len(tp) == 0:
            continue
        if num_gt == 0:
            aps.append(0.0)
            continue

        # 计算 AP（简化版 11-point interpolation）
        tp_cum = torch.tensor(tp).cumsum(0)
        fp_cum = torch.tensor(fp).cumsum(0)
        recalls = tp_cum / max(num_gt, 1)
        precisions = tp_cum / (tp_cum + fp_cum + 1e-7)

        ap = 0.0
        for r in torch.linspace(0, 1.0, 11):
            mask = recalls >= r
            if mask.any():
                ap += precisions[mask].max().item()
        aps.append(ap / 11.0)

    mAP = sum(aps) / max(len(aps), 1)
    # 首次验证时打印诊断信息
    if total_gt > 0:
        print(f"  [诊断] GT框: {total_gt} | 预测框(score>0.05): {total_pd} | 预测/GT: {total_pd/total_gt:.1f}x")
    return mAP


def box_iou(boxes1, boxes2):
    """计算两组 bbox 的 IoU"""
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    return inter / (union + 1e-7)


def train_detection():
    """训练检测模型"""
    set_seed(RANDOM_SEED)

    print("\n" + "=" * 60)
    print("PCB缺陷检测 — Faster R-CNN (ResNet50-FPN)")
    print("=" * 60)

    # DataLoader
    num_workers = 0 if os.name == "nt" else 4
    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=DET_BATCH_SIZE, num_workers=num_workers)

    # 模型
    model = create_model(num_classes=len(CLASS_NAMES) + 1, pretrained=True)
    params_M = count_parameters(model)
    print(f"模型参数量: {params_M:.2f}M  |  设备: {DEVICE}")
    model.to(DEVICE)

    # 优化器
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=DET_LR, momentum=0.9, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=DET_LR_STEP, gamma=DET_LR_GAMMA)
    warmup = warmup_lr_scheduler(optimizer, warmup_iters=min(1000, len(train_loader)),
                                 warmup_factor=1.0 / 3)

    history = {"train_loss": [], "val_mAP": []}
    best_mAP = 0.0
    n_no_improve = 0

    print(f"{'='*70}")
    print(f"  Batch: {DET_BATCH_SIZE}  |  Epochs: {DET_NUM_EPOCHS}  |  LR: {DET_LR}  |  早停: {DET_EARLY_STOP}")
    print(f"{'='*70}")

    t0 = time.time()
    for epoch in range(1, DET_NUM_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{DET_NUM_EPOCHS}",
                     bar_format="{l_bar}{bar:30}{r_bar}", ncols=110)

        for batch_idx, (images, targets) in enumerate(pbar):
            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            if not torch.isfinite(losses):
                continue

            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            warmup.step()

            epoch_loss += losses.item()
            avg_loss = epoch_loss / (batch_idx + 1)

            # 显示各组件 loss
            comps = {k: f"{v.item():.3f}" for k, v in loss_dict.items()}
            comps["total"] = f"{avg_loss:.4f}"
            pbar.set_postfix_str(
                f"cls={comps.get('loss_classifier','-')} "
                f"box={comps.get('loss_box_reg','-')} "
                f"obj={comps.get('loss_objectness','-')} "
                f"rpn={comps.get('loss_rpn_box_reg','-')}"
            )

        epoch_loss /= len(train_loader)
        scheduler.step()

        # 验证
        val_mAP = evaluate(model, val_loader, DEVICE)
        history["train_loss"].append(round(epoch_loss, 4))
        history["val_mAP"].append(round(val_mAP, 4))

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"  Epoch {epoch:3d}/{DET_NUM_EPOCHS} │ loss: {epoch_loss:.4f}  "
              f"mAP@0.5: {val_mAP:.4f}  lr: {lr_now:.2e}"
              f"{' │ ★ best' if val_mAP > best_mAP else ''}")

        if val_mAP > best_mAP:
            best_mAP = val_mAP
            torch.save(model.state_dict(),
                       os.path.join(MODELS_DIR, "detection_faster_rcnn_best.pth"))
            n_no_improve = 0
        else:
            n_no_improve += 1
            if n_no_improve >= DET_EARLY_STOP:
                print(f"  早停触发于 Epoch {epoch}")
                break

    train_time = time.time() - t0
    print(f"训练完成 | 耗时: {train_time:.1f}s | 最佳 mAP@0.5: {best_mAP:.4f}")

    # 保存训练历史
    with open(os.path.join(LOGS_DIR, "detection_history.json"), "w") as f:
        json.dump({
            "train_time_s": round(train_time, 1),
            "best_mAP": round(best_mAP, 4),
            "params_M": round(params_M, 2),
            "history": history,
        }, f, indent=2)

    return model, history


if __name__ == "__main__":
    train_detection()
