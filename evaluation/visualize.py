"""
可视化模块 - 训练曲线、混淆矩阵、模型对比图表
Author: CleartoCloudy
"""
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 非交互后端，兼容无GUI环境
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CLASS_NAMES, FIGURES_DIR, CM_DIR, LOGS_DIR

# 中文字体设置
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def plot_confusion_matrix(cm, model_name, save_path=None):
    """绘制单个模型的混淆矩阵"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                cbar_kws={"shrink": 0.8})
    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(CM_DIR, f"{model_name.replace('+', '_')}_cm.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


def plot_all_confusion_matrices(all_results):
    """绘制所有模型的混淆矩阵"""
    for model_name, metrics in all_results.items():
        cm = metrics.get("confusion_matrix")
        if cm is not None:
            if isinstance(cm, list):
                cm = np.array(cm)
            plot_confusion_matrix(cm, model_name)
            print(f"  混淆矩阵已保存: {model_name}")


def plot_training_curves(model_name=None):
    """
    绘制训练曲线
    如指定model_name则只绘制该模型，否则绘制所有深度学习模型的训练曲线
    """
    log_dir = LOGS_DIR
    history_files = [f for f in os.listdir(log_dir) if f.endswith("_history.json")]

    if not history_files:
        print("未找到训练历史文件")
        return

    for hist_file in history_files:
        name = hist_file.replace("_history.json", "")
        if model_name and model_name not in name:
            continue

        with open(os.path.join(log_dir, hist_file), "r") as f:
            history = json.load(f)

        # 跳过传统方法的 history 文件（只有 train_time_s，没有训练曲线数据）
        if "train_loss" not in history:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 损失曲线
        axes[0].plot(history["train_loss"], label="Train Loss", linewidth=2)
        axes[0].plot(history["val_loss"], label="Val Loss", linewidth=2)
        axes[0].set_title(f"{name} - Loss Curve")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 准确率曲线
        axes[1].plot(history["train_acc"], label="Train Acc", linewidth=2)
        axes[1].plot(history["val_acc"], label="Val Acc", linewidth=2)
        axes[1].set_title(f"{name} - Accuracy Curve")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(FIGURES_DIR, f"{name}_training_curves.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  训练曲线已保存: {save_path}")


def plot_model_comparison(results_summary, metric="accuracy"):
    """绘制模型性能对比柱状图"""
    if not results_summary:
        print("无结果可用于对比")
        return

    names = list(results_summary.keys())
    values = [results_summary[n].get(metric, 0) for n in names]
    colors = ["#3498db" if results_summary[n].get("type") == "traditional" else "#e74c3c"
              for n in names]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(names)), values, color=colors, edgecolor="white", linewidth=1.2)

    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{val:.4f}", ha="center", fontsize=11, fontweight="bold")

    plt.xticks(range(len(names)), names, rotation=20, ha="right")
    plt.ylabel(metric.capitalize())
    plt.title(f"Model Comparison - {metric.capitalize()}")
    plt.ylim(0, max(values) * 1.15 if max(values) > 0 else 1)
    plt.grid(axis="y", alpha=0.3)

    # 图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#3498db", label="Traditional Methods"),
        Patch(facecolor="#e74c3c", label="Deep Learning Methods"),
    ]
    plt.legend(handles=legend_elements)

    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, f"comparison_{metric}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  对比图已保存: {save_path}")


def plot_comprehensive_comparison(results_summary):
    """综合对比图（含准确率、F1、参数量的多子图）"""
    if not results_summary:
        return

    names = list(results_summary.keys())
    x = np.arange(len(names))
    width = 0.35

    metrics_list = [
        ("accuracy", "Accuracy"),
        ("f1", "F1-Score"),
        ("precision", "Precision"),
        ("recall", "Recall"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    colors = ["#3498db" if results_summary[n].get("type") == "traditional" else "#e74c3c"
              for n in names]

    for ax, (metric, title) in zip(axes.flatten(), metrics_list):
        values = [results_summary[n].get(metric, 0) for n in names]
        bars = ax.bar(names, values, color=colors, edgecolor="white", linewidth=1.2)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")
        ax.set_title(title)
        ax.set_ylim(0, max(values) * 1.2 if max(values) > 0 else 1)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Comprehensive Model Performance Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, "comprehensive_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  综合对比图已保存: {save_path}")


def plot_efficiency_comparison(results_summary):
    """模型效率对比（参数量 vs 推理时间）"""
    dl_results = {k: v for k, v in results_summary.items() if v.get("type") == "deep_learning"}
    if not dl_results:
        return

    names = list(dl_results.keys())
    params = [dl_results[n].get("params_M", 0) for n in names]
    times = [dl_results[n].get("inference_time_ms", 0) for n in names]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    bars = ax1.bar(names, params, color="#3498db", alpha=0.7, label="Parameters (M)")
    ax1.set_ylabel("Parameters (Millions)")
    ax1.set_title("Model Efficiency Comparison")

    ax2 = ax1.twinx()
    ax2.plot(names, times, "ro-", linewidth=2, markersize=8, label="Inference Time (ms/batch)")
    ax2.set_ylabel("Inference Time (ms/batch)")

    for i, (p, t) in enumerate(zip(params, times)):
        ax1.text(i, p + 0.1, f"{p:.1f}M", ha="center", fontsize=9)
        ax2.text(i, t + 0.3, f"{t:.1f}ms", ha="center", fontsize=9, color="red")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, "efficiency_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  效率对比图已保存: {save_path}")


def generate_all_visualizations(all_results=None):
    """生成所有可视化图表"""
    print("\n" + "=" * 60)
    print("生成可视化图表")
    print("=" * 60)

    # 1. 训练曲线
    print("\n[1] 训练曲线...")
    plot_training_curves()

    # 2. 混淆矩阵
    if all_results:
        print("\n[2] 混淆矩阵...")
        plot_all_confusion_matrices(all_results)

        # 3. 模型对比图
        print("\n[3] 准确率对比...")
        plot_model_comparison(all_results, "accuracy")

        print("\n[4] F1对比...")
        plot_model_comparison(all_results, "f1")

        print("\n[5] 综合对比...")
        plot_comprehensive_comparison(all_results)

        print("\n[6] 效率对比...")
        plot_efficiency_comparison(all_results)

    print("\n所有图表已保存至 results/figures/ 和 results/confusion_matrix/")


def main():
    # 尝试加载评估结果
    summary_path = os.path.join(LOGS_DIR, "all_results_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            all_results = json.load(f)
        # 转换混淆矩阵为numpy
        for k in all_results:
            if "confusion_matrix" in all_results[k] and isinstance(all_results[k]["confusion_matrix"], list):
                all_results[k]["confusion_matrix"] = np.array(all_results[k]["confusion_matrix"])
    else:
        all_results = None

    generate_all_visualizations(all_results)


if __name__ == "__main__":
    main()
