"""
实验报告生成器 - 汇总所有实验数据，生成 Markdown + JSON 格式的完整报告
每运行一次生成一个带时间戳的新文件，文件名直观易读
Author: CleartoCloudy
"""
import os
import sys
import json
import time
import platform
from datetime import datetime
from collections import OrderedDict

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATASET_PROCESSED, CLASS_NAMES, LOGS_DIR, MODELS_DIR, DEVICE,
                    BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, IMG_SIZE_TRAD, IMG_SIZE_DL,
                    RANDOM_SEED)

# ---- 报告输出路径 ----
REPORT_DIR = os.path.join(os.path.dirname(LOGS_DIR), "reports")
os.makedirs(REPORT_DIR, exist_ok=True)


def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_readable_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ==================== 数据收集 ====================

def count_dataset():
    """统计数据集各类别样本数"""
    stats = {"train": {}, "val": {}, "test": {}, "train_augmented": {}}
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(DATASET_PROCESSED, split)
        if not os.path.exists(split_dir):
            continue
        for cls in CLASS_NAMES:
            cls_dir = os.path.join(split_dir, cls)
            if os.path.exists(cls_dir):
                stats[split][cls] = len([f for f in os.listdir(cls_dir)
                                         if f.endswith((".jpg", ".png", ".bmp"))])
            else:
                stats[split][cls] = 0

    # 增强后的训练集
    aug_dir = os.path.join(DATASET_PROCESSED, "train_augmented")
    if os.path.exists(aug_dir):
        for cls in CLASS_NAMES:
            cls_dir = os.path.join(aug_dir, cls)
            if os.path.exists(cls_dir):
                stats["train_augmented"][cls] = len([f for f in os.listdir(cls_dir)
                                                      if f.endswith((".jpg", ".png", ".bmp"))])

    return stats


def load_model_results():
    """
    加载所有可用模型的测试/评估结果
    来源: all_results_summary.json + 各模型 _result.json
    返回 OrderedDict
    """
    results = OrderedDict()

    # 实验顺序
    exp_order = ["HOG+SVM", "LBP+SVM", "ResNet18", "MobileNetV3",
                 "ResNet18+SE", "ResNet18+CBAM"]
    result_files = {
        "HOG+SVM": None,
        "LBP+SVM": None,
        "ResNet18": "resnet18_result.json",
        "MobileNetV3": "mobilenetv3_result.json",
        "ResNet18+SE": "resnet18_se_result.json",
        "ResNet18+CBAM": "resnet18_cbam_result.json",
    }

    # 优先从 all_results_summary.json 加载
    summary_path = os.path.join(LOGS_DIR, "all_results_summary.json")
    summary_data = {}
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            summary_data = json.load(f)

    for name in exp_order:
        entry = {"name": name, "available": False}

        # 从 summary 获取
        if name in summary_data:
            s = summary_data[name]
            entry.update({
                "available": True,
                "type": s.get("type", "unknown"),
                "accuracy": s.get("accuracy"),
                "precision": s.get("precision"),
                "recall": s.get("recall"),
                "f1": s.get("f1"),
                "params_M": s.get("params_M", 0),
                "inference_time_ms": s.get("inference_time_ms", 0),
            })
            if "confusion_matrix" in s and s["confusion_matrix"]:
                entry["cm"] = np.array(s["confusion_matrix"])
            # 判断是否为最佳模型
            entry["is_best_acc"] = False
            entry["is_best_f1"] = False

        # 补充深度学习的训练历史
        hist_file = result_files.get(name)
        if hist_file:
            hist_path = os.path.join(LOGS_DIR, hist_file)
            if os.path.exists(hist_path):
                with open(hist_path, "r") as f:
                    dl_result = json.load(f)
                entry["train_acc"] = dl_result.get("train_acc")
                entry["val_acc"] = dl_result.get("val_acc")
                entry["train_time_s"] = dl_result.get("train_time_s")
                if "history" in dl_result:
                    entry["epochs_trained"] = len(dl_result["history"]["train_loss"])
                    entry["final_train_loss"] = dl_result["history"]["train_loss"][-1]
                    entry["final_val_loss"] = dl_result["history"]["val_loss"][-1]

        # 传统方法补充训练信息
        if name in ["HOG+SVM", "LBP+SVM"] and entry.get("available"):
            model_type_key = "hog_svm" if name == "HOG+SVM" else "lbp_svm"
            hist_path = os.path.join(LOGS_DIR, f"{model_type_key}_history.json")
            if os.path.exists(hist_path):
                with open(hist_path, "r") as f:
                    trad_hist = json.load(f)
                entry["train_time_s"] = trad_hist.get("train_time_s")

        results[name] = entry

    # 标注最佳模型
    avail = [(k, v) for k, v in results.items() if v.get("available")]
    if avail:
        best_acc = max(avail, key=lambda x: x[1].get("accuracy", 0))
        best_f1 = max(avail, key=lambda x: x[1].get("f1", 0))
        results[best_acc[0]]["is_best_acc"] = True
        results[best_f1[0]]["is_best_f1"] = True

    return results


def load_detection_results():
    """加载检测模型结果"""
    info = {"available": False}
    hist_path = os.path.join(LOGS_DIR, "detection_history.json")
    if os.path.exists(hist_path):
        with open(hist_path, "r") as f:
            hist = json.load(f)
        info.update({
            "available": True,
            "params_M": hist.get("params_M", 0),
            "train_time_s": hist.get("train_time_s", 0),
            "best_mAP": hist.get("best_mAP", 0),
        })
    result_path = os.path.join(LOGS_DIR, "detection_result.json")
    if os.path.exists(result_path):
        with open(result_path, "r") as f:
            r = json.load(f)
        info["test_mAP"] = r.get("mAP50", 0)
    return info


def get_env_info():
    """收集运行环境信息"""
    info = {
        "os": platform.system(),
        "os_version": platform.version(),
        "python": platform.python_version(),
        "device": DEVICE,
    }
    try:
        import torch
        info["pytorch"] = torch.__version__
        if torch.cuda.is_available():
            info["cuda"] = torch.version.cuda
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory"] = f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
    except ImportError:
        info["pytorch"] = "not installed"

    try:
        import cv2
        info["opencv"] = cv2.__version__
    except ImportError:
        pass

    info["config"] = {
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "img_size_trad": IMG_SIZE_TRAD,
        "img_size_dl": IMG_SIZE_DL,
        "random_seed": RANDOM_SEED,
    }
    return info


# ==================== Markdown 报告生成 ====================

def _md_header(title, level=1):
    return f"{'#' * level} {title}\n"


def _md_table(headers, rows, align=None):
    """生成 Markdown 表格"""
    lines = []
    lines.append("| " + " | ".join(str(h) for h in headers) + " |")
    if align is None:
        align = ["---"] * len(headers)
    lines.append("| " + " | ".join(align) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines) + "\n"


def _status_icon(available):
    return "OK" if available else "N/A"


def _best_badge(is_best):
    return " **★最佳**" if is_best else ""


def _format_pct(val):
    if val is None:
        return "—"
    return f"{val * 100:.2f}%"


def generate_markdown_report(results, dataset_stats, env_info, timestamp_readable):
    """生成完整 Markdown 报告内容"""

    # 找到最佳模型
    best_acc_name = None
    best_f1_name = None
    for name, r in results.items():
        if r.get("is_best_acc"):
            best_acc_name = name
        if r.get("is_best_f1"):
            best_f1_name = name

    lines = []
    lines.append(f"# PCB 缺陷分类 — 实验报告\n")
    lines.append(f"> 生成时间: {timestamp_readable}  |  Author: CleartoCloudy\n")
    lines.append("---\n")

    # ========== 一、实验环境 ==========
    lines.append(_md_header("一、实验环境", 2))
    lines.append(f"- **操作系统**: {env_info['os']} {env_info.get('os_version', '')}")
    lines.append(f"- **Python 版本**: {env_info['python']}")
    lines.append(f"- **PyTorch 版本**: {env_info.get('pytorch', 'N/A')}")
    lines.append(f"- **计算设备**: {env_info['device']}")
    if "gpu_name" in env_info:
        lines.append(f"- **GPU**: {env_info['gpu_name']} ({env_info.get('gpu_memory', 'N/A')})")
    if "cuda" in env_info:
        lines.append(f"- **CUDA**: {env_info['cuda']}")
    cfg = env_info.get("config", {})
    lines.append(f"- **Batch Size**: {cfg.get('batch_size')}")
    lines.append(f"- **最大 Epochs**: {cfg.get('num_epochs')}")
    lines.append(f"- **学习率**: {cfg.get('learning_rate')}")
    lines.append(f"- **传统方法图像尺寸**: {cfg.get('img_size_trad')}×{cfg.get('img_size_trad')}")
    lines.append(f"- **深度学习图像尺寸**: {cfg.get('img_size_dl')}×{cfg.get('img_size_dl')}")
    lines.append(f"- **随机种子**: {cfg.get('random_seed')}")
    lines.append("")

    # ========== 二、数据集统计 ==========
    lines.append(_md_header("二、数据集统计", 2))

    splits_available = [s for s in ["train", "train_augmented", "val", "test"]
                        if s in dataset_stats and any(v > 0 for v in dataset_stats[s].values())]
    if splits_available:
        headers = ["类别"] + splits_available + (["总计"] if len(splits_available) > 1 else [])
        rows = []
        totals = {s: 0 for s in splits_available}
        for cls in CLASS_NAMES:
            row = [cls]
            row_total = 0
            for s in splits_available:
                cnt = dataset_stats[s].get(cls, 0)
                row.append(str(cnt))
                totals[s] += cnt
                row_total += cnt
            if len(splits_available) > 1:
                row.append(str(row_total))
            rows.append(row)
        # 总计行
        total_row = ["**总计**"]
        grand_total = 0
        for s in splits_available:
            total_row.append(f"**{totals[s]}**")
            grand_total += totals[s]
        if len(splits_available) > 1:
            total_row.append(f"**{grand_total}**")
        rows.append(total_row)

        lines.append(_md_table(headers, rows))
    lines.append("")

    # ========== 三、模型性能总览 ==========
    lines.append(_md_header("三、模型性能总览", 2))
    lines.append("> ★ 标注为该项指标的最佳模型\n")

    overview_headers = ["实验", "方法", "类型", "Accuracy", "Precision", "Recall",
                        "F1-Score", "参数量(M)", "推理时间(ms)"]
    overview_rows = []
    exp_labels = ["实验一", "实验二", "实验三", "实验四", "实验五", "实验五(变体)"]
    exp_idx = 0

    for name, r in results.items():
        if not r.get("available"):
            overview_rows.append([exp_labels[exp_idx] if exp_idx < len(exp_labels) else "—",
                                  name, "—", "—", "—", "—", "—", "—", "—"])
        else:
            acc_str = _format_pct(r.get("accuracy")) + _best_badge(r.get("is_best_acc"))
            f1_str = _format_pct(r.get("f1")) + _best_badge(r.get("is_best_f1"))
            overview_rows.append([
                exp_labels[exp_idx] if exp_idx < len(exp_labels) else "—",
                f"**{name}**" if (r.get("is_best_acc") or r.get("is_best_f1")) else name,
                "传统方法" if r.get("type") == "traditional" else "深度学习",
                acc_str,
                _format_pct(r.get("precision")),
                _format_pct(r.get("recall")),
                f1_str,
                f"{r.get('params_M', 0):.2f}" if r.get("type") == "deep_learning" else "—",
                f"{r.get('inference_time_ms', 0):.1f}" if r.get("inference_time_ms", 0) > 0 else "—",
            ])
        exp_idx += 1

    lines.append(_md_table(overview_headers, overview_rows))
    lines.append("")

    # ========== 四、各模型详细结果 ==========
    lines.append(_md_header("四、各模型详细结果", 2))

    exp_num = 1
    for name, r in results.items():
        if name == "ResNet18+CBAM" and results.get("ResNet18+SE", {}).get("available"):
            label = "实验五(变体)"
        else:
            label = f"实验{exp_num}"
        exp_num += 1

        lines.append(_md_header(f"{label}: {name}", 3))

        if not r.get("available"):
            lines.append("> 该模型实验未完成或结果不可用\n")
            continue

        # 指标卡片
        lines.append(f"- **类型**: {'传统模式识别' if r.get('type') == 'traditional' else '深度学习'}")
        lines.append(f"- **Accuracy**: {_format_pct(r.get('accuracy'))}")
        lines.append(f"- **Precision**: {_format_pct(r.get('precision'))}")
        lines.append(f"- **Recall**: {_format_pct(r.get('recall'))}")
        lines.append(f"- **F1-Score**: {_format_pct(r.get('f1'))}")

        if r.get("type") == "deep_learning":
            lines.append(f"- **参数量**: {r.get('params_M', 0):.2f} M")
            lines.append(f"- **推理时间**: {r.get('inference_time_ms', 0):.2f} ms/batch")

        if r.get("train_time_s"):
            minutes = r["train_time_s"] / 60
            lines.append(f"- **训练耗时**: {minutes:.1f} 分钟 ({r['train_time_s']:.0f} 秒)")

        if r.get("epochs_trained"):
            lines.append(f"- **训练轮次**: {r['epochs_trained']} epochs")
            if r.get("final_train_loss"):
                lines.append(f"- **最终训练 Loss**: {r['final_train_loss']:.4f}")
            if r.get("final_val_loss"):
                lines.append(f"- **最终验证 Loss**: {r['final_val_loss']:.4f}")

        # 训练过程中的最佳/最终指标
        if r.get("train_acc"):
            lines.append(f"- **训练集 Acc**: {_format_pct(r.get('train_acc'))}")
        if r.get("val_acc"):
            lines.append(f"- **验证集 Acc**: {_format_pct(r.get('val_acc'))}")

        # 混淆矩阵
        if r.get("cm") is not None:
            cm = r["cm"]
            lines.append(f"\n**混淆矩阵:**\n")
            # 表头
            header = [""] + [f"{CLASS_NAMES[i]}(预)" for i in range(len(CLASS_NAMES))]
            rows = []
            for i in range(len(CLASS_NAMES)):
                row = [f"{CLASS_NAMES[i]}(真)"] + [str(cm[i][j]) for j in range(len(CLASS_NAMES))]
                rows.append(row)
            lines.append(_md_table(header, rows))

            # 各类别准确率
            lines.append(f"\n**各类别准确率:**\n")
            cls_acc_rows = []
            for i in range(len(CLASS_NAMES)):
                row_sum = cm[i].sum()
                if row_sum > 0:
                    cls_acc = cm[i][i] / row_sum
                    cls_acc_rows.append([CLASS_NAMES[i], _format_pct(cls_acc), f"{cm[i][i]}/{row_sum}"])
            lines.append(_md_table(["类别", "准确率", "正确/总数"], cls_acc_rows))

        lines.append("")

    # ========== 五、模型效率对比 ==========
    lines.append(_md_header("五、模型效率对比", 2))

    dl_models = [(name, r) for name, r in results.items()
                 if r.get("available") and r.get("type") == "deep_learning"]

    if dl_models:
        lines.append("> 仅对比深度学习模型（传统方法参数量和推理时间不计）\n")
        eff_headers = ["模型", "参数量(M)", "推理时间(ms/batch)", "训练耗时(分钟)", "Accuracy"]
        eff_rows = []
        for name, r in dl_models:
            train_min = f"{r.get('train_time_s', 0) / 60:.1f}" if r.get("train_time_s") else "—"
            eff_rows.append([
                name,
                f"{r.get('params_M', 0):.2f}",
                f"{r.get('inference_time_ms', 0):.2f}",
                train_min,
                _format_pct(r.get("accuracy")),
            ])
        lines.append(_md_table(eff_headers, eff_rows))

        # 效率评分: Acc / params * inference_time 的简化版本
        if len(dl_models) >= 2:
            lines.append(f"\n**效率分析:**")
            # 找最轻量和最高精度
            lightest = min(dl_models, key=lambda x: x[1].get("params_M", float("inf")))
            fastest = min(dl_models, key=lambda x: x[1].get("inference_time_ms", float("inf")))
            lines.append(f"- 最轻量: **{lightest[0]}** ({lightest[1].get('params_M', 0):.2f}M)")
            lines.append(f"- 推理最快: **{fastest[0]}** ({fastest[1].get('inference_time_ms', 0):.2f}ms/batch)")
            lines.append(f"- 精度最高: **{best_acc_name}** ({results.get(best_acc_name, {}).get('accuracy', 0) * 100:.2f}%)")
            lines.append("")
    else:
        lines.append("> 无深度学习模型结果\n")
    lines.append("")

    # ========== 5.5 目标检测（可选） ==========
    det_info = load_detection_results()
    if det_info["available"]:
        lines.append(_md_header("五、目标检测 (Faster R-CNN)", 2))
        lines.append(f"- **模型**: Faster R-CNN (ResNet50-FPN)")
        lines.append(f"- **参数量**: {det_info['params_M']:.2f} M")
        lines.append(f"- **训练耗时**: {det_info.get('train_time_s', 0) / 60:.1f} 分钟")
        lines.append(f"- **最佳 mAP@0.5**: {det_info['best_mAP']:.4f}")
        if det_info.get("test_mAP"):
            lines.append(f"- **测试集 mAP@0.5**: {det_info['test_mAP']:.4f}")
        lines.append("")
        lines.append("> 检测任务输入完整 PCB 大图（640×640），输出所有缺陷的 bounding box + 类别。")
        lines.append("> 与分类任务互补：分类判断已定位缺陷的类型，检测从大图中找出全部缺陷的位置并分类。")
        lines.append("")

    # ========== 六、总结 ==========
    lines.append(_md_header("六、总结", 2))

    avail_count = sum(1 for r in results.values() if r.get("available"))
    trad_count = sum(1 for r in results.values()
                     if r.get("available") and r.get("type") == "traditional")
    dl_count = avail_count - trad_count

    lines.append(f"本次实验共完成 **{avail_count}** 个模型的训练与评估"
                 f"（传统方法 {trad_count} 个，深度学习方法 {dl_count} 个）。")

    if best_acc_name:
        lines.append(f"\n- **准确率最高**: {best_acc_name}"
                     f"（{_format_pct(results[best_acc_name].get('accuracy'))}）")
    if best_f1_name:
        lines.append(f"- **F1-Score 最高**: {best_f1_name}"
                     f"（{_format_pct(results[best_f1_name].get('f1'))}）")

    if dl_count >= 2 and trad_count >= 1:
        # 深度学习 vs 传统方法对比
        dl_accs = [r.get("accuracy", 0) for n, r in results.items()
                   if r.get("available") and r.get("type") == "deep_learning"]
        trad_accs = [r.get("accuracy", 0) for n, r in results.items()
                     if r.get("available") and r.get("type") == "traditional"]
        if dl_accs and trad_accs:
            lines.append(f"\n- 深度学习模型平均准确率: **{np.mean(dl_accs) * 100:.2f}%**")
            lines.append(f"- 传统方法平均准确率: **{np.mean(trad_accs) * 100:.2f}%**")
            if np.mean(dl_accs) > np.mean(trad_accs):
                lines.append(f"- 深度学习相比传统方法准确率提升约 **{(np.mean(dl_accs) - np.mean(trad_accs)) * 100:.1f}** 个百分点")

    lines.append(f"\n---\n")
    lines.append(f"*报告由 PCBclassification 实验系统自动生成 | {timestamp_readable}*")
    lines.append("")

    return "\n".join(lines)


# ==================== JSON 报告 ====================

def generate_json_report(results, dataset_stats, env_info, timestamp_readable):
    """生成结构化 JSON 报告"""
    report = OrderedDict()
    report["title"] = "PCB缺陷分类实验报告"
    report["generated_at"] = timestamp_readable
    report["author"] = "CleartoCloudy"
    report["environment"] = env_info
    report["dataset_statistics"] = dataset_stats

    models = OrderedDict()
    for name, r in results.items():
        entry = OrderedDict()
        entry["available"] = r.get("available", False)
        if r.get("available"):
            entry["type"] = r.get("type")
            entry["accuracy"] = r.get("accuracy")
            entry["precision"] = r.get("precision")
            entry["recall"] = r.get("recall")
            entry["f1"] = r.get("f1")
            entry["params_M"] = r.get("params_M")
            entry["inference_time_ms"] = r.get("inference_time_ms")
            entry["train_time_s"] = r.get("train_time_s")
            entry["epochs_trained"] = r.get("epochs_trained")
            entry["train_acc"] = r.get("train_acc")
            entry["val_acc"] = r.get("val_acc")
            entry["is_best_acc"] = r.get("is_best_acc", False)
            entry["is_best_f1"] = r.get("is_best_f1", False)
            if r.get("cm") is not None:
                entry["confusion_matrix"] = r["cm"].tolist()
        models[name] = entry
    report["models"] = models

    # 目标检测结果
    det = load_detection_results()
    if det["available"]:
        report["detection"] = OrderedDict([
            ("model", "Faster R-CNN (ResNet50-FPN)"),
            ("mAP50_val", det["best_mAP"]),
            ("mAP50_test", det.get("test_mAP")),
            ("params_M", det["params_M"]),
            ("train_time_s", det.get("train_time_s")),
        ])

    return report


# ==================== 主函数 ====================

def generate_report():
    """
    收集所有数据，生成 Markdown 和 JSON 两份报告
    返回: (md_path, json_path)
    """
    ts = get_timestamp()
    ts_readable = get_readable_time()

    print("\n" + "=" * 70)
    print("生成实验报告")
    print("=" * 70)

    # 收集数据
    print("[1/4] 收集数据集统计...")
    dataset_stats = count_dataset()

    print("[2/4] 收集环境信息...")
    env_info = get_env_info()

    print("[3/4] 收集模型评估结果...")
    results = load_model_results()

    print("[4/4] 生成报告文件...")

    # 生成 Markdown
    md_content = generate_markdown_report(results, dataset_stats, env_info, ts_readable)
    md_filename = f"experiment_report_{ts}.md"
    md_path = os.path.join(REPORT_DIR, md_filename)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    # 生成 JSON
    json_report = generate_json_report(results, dataset_stats, env_info, ts_readable)
    json_filename = f"experiment_report_{ts}.json"
    json_path = os.path.join(REPORT_DIR, json_filename)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_report, f, indent=2, ensure_ascii=False)

    # 同时保存一份 latest 副本（方便快速查看最新结果）
    latest_md = os.path.join(REPORT_DIR, "latest_report.md")
    with open(latest_md, "w", encoding="utf-8") as f:
        f.write(md_content)
    latest_json = os.path.join(REPORT_DIR, "latest_report.json")
    with open(latest_json, "w", encoding="utf-8") as f:
        json.dump(json_report, f, indent=2, ensure_ascii=False)

    print(f"\n报告已生成:")
    print(f"  Markdown: {md_path}")
    print(f"  JSON:     {json_path}")
    print(f"  最新副本: {latest_md}")
    print(f"  最新副本: {latest_json}")

    # 输出简要汇总
    avail = [(n, r) for n, r in results.items() if r.get("available")]
    if avail:
        print(f"\n快速汇总 ({len(avail)} 个模型):")
        print(f"  {'方法':<20s} {'Acc':>8s} {'F1':>8s} {'Params':>8s}")
        print(f"  {'-'*44}")
        for name, r in avail:
            params_str = f"{r.get('params_M', 0):.1f}M" if r.get("type") == "deep_learning" else "—"
            print(f"  {name:<20s} {_format_pct(r.get('accuracy')):>8s} "
                  f"{_format_pct(r.get('f1')):>8s} {params_str:>8s}")

    return md_path, json_path


def main():
    generate_report()


if __name__ == "__main__":
    main()
