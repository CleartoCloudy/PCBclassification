#!/usr/bin/env python
"""
PCB缺陷分类系统 - 统一入口
Author: CleartoCloudy

用法:
  python main.py preprocess   # 数据预处理（裁剪+CLAHE+增强）
  python main.py train all    # 训练所有模型
  python main.py train hog_svm  # 训练指定模型
  python main.py evaluate     # 评估所有模型 + 生成实验报告
  python main.py visualize    # 生成可视化图表
  python main.py report       # 单独生成实验报告
  python main.py run          # 一键运行完整流程
  python main.py predict --model resnet18 --image <path>  # 单图预测
"""
import os
import sys
import argparse

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def cmd_preprocess(args):
    """数据预处理流程"""
    print("\n" + "=" * 70)
    print("步骤 1/3: 数据集裁剪与划分")
    print("=" * 70)
    from preprocess.crop_dataset import main as crop_main
    crop_main()

    print("\n" + "=" * 70)
    print("步骤 2/3: CLAHE 图像增强")
    print("=" * 70)
    from preprocess.clahe import main as clahe_main
    clahe_main()

    print("\n" + "=" * 70)
    print("步骤 3/3: 数据增强")
    print("=" * 70)
    from preprocess.augment import main as augment_main
    augment_main()

    print("\n数据预处理完成！")


def cmd_train(args):
    """模型训练"""
    model_name = args.model

    # 传统方法
    if model_name == "hog_svm" or model_name == "all":
        from traditional.hog_svm import main as hog_main
        hog_main()

    if model_name == "lbp_svm" or model_name == "all":
        from traditional.lbp_svm import main as lbp_main
        lbp_main()

    # 深度学习方法
    dl_models = ["resnet18", "mobilenetv3", "resnet18_se", "resnet18_cbam"]
    for m in dl_models:
        if model_name == m or model_name == "all":
            from deep_learning.train import run_dl_experiment
            run_dl_experiment(m)

    print("\n训练完成！")


def cmd_evaluate(args):
    """模型评估 + 生成实验报告"""
    from evaluation.evaluate import evaluate_all_models
    evaluate_all_models()

    # 评估完成后自动生成报告
    from evaluation.report import generate_report
    generate_report()


def cmd_visualize(args):
    """生成可视化图表"""
    from evaluation.visualize import generate_all_visualizations
    import json
    import numpy as np
    from config import LOGS_DIR

    summary_path = os.path.join(LOGS_DIR, "all_results_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            all_results = json.load(f)
        for k in all_results:
            if "confusion_matrix" in all_results[k] and isinstance(all_results[k]["confusion_matrix"], list):
                all_results[k]["confusion_matrix"] = np.array(all_results[k]["confusion_matrix"])
    else:
        all_results = None

    generate_all_visualizations(all_results)


def cmd_report(args):
    """单独生成实验报告（不重新评估，基于已有结果）"""
    from evaluation.report import generate_report
    generate_report()


def cmd_predict(args):
    """单图/批量预测"""
    from deep_learning.predict import load_model, predict_single, predict_batch
    from config import CLASS_NAMES

    model_info = load_model(args.model)
    print(f"模型: {args.model}  |  类型: {model_info['type']}")

    if os.path.isdir(args.image):
        image_paths = [
            os.path.join(args.image, f) for f in os.listdir(args.image)
            if f.endswith((".jpg", ".png", ".bmp"))
        ]
        results = predict_batch(model_info, image_paths)
        for r in results:
            if "error" not in r:
                print(f"  {os.path.basename(r['path'])} -> {r['class_name']} ({r['confidence']:.4f})")
    else:
        name, cid, probs = predict_single(model_info, args.image, return_probs=True)
        print(f"图像: {args.image}")
        print(f"预测: {name} (置信度: {probs[cid]:.4f})")
        print("各类别概率:")
        for i, cn in enumerate(CLASS_NAMES):
            bar = "█" * int(probs[i] * 40)
            print(f"  {cn:20s}  {probs[i]:.4f}  {bar}")


def cmd_run(args):
    """一键运行完整流程"""
    print("=" * 70)
    print("PCB缺陷分类系统 - 一键运行完整流程")
    print("=" * 70)

    # 1. 数据预处理
    cmd_preprocess(args)

    # 2. 训练所有模型
    args.model = "all"
    cmd_train(args)

    # 3. 评估
    cmd_evaluate(args)

    # 4. 可视化
    cmd_visualize(args)

    # 5. 生成实验报告
    from evaluation.report import generate_report
    generate_report()

    print("\n" + "=" * 70)
    print("完整流程运行完毕！")
    print(f"模型文件: results/models/")
    print(f"评估结果: results/logs/")
    print(f"可视化图表: results/figures/ 和 results/confusion_matrix/")
    print(f"实验报告: results/reports/")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="PCB缺陷分类系统 - 基于多种模式识别方法的比较研究",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py preprocess              # 数据预处理
  python main.py train all               # 训练所有模型
  python main.py train resnet18          # 只训练ResNet18
  python main.py evaluate                # 评估所有模型
  python main.py visualize               # 生成可视化图表
  python main.py run                     # 一键运行完整流程
  python main.py predict --model resnet18 --image defect.jpg
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # preprocess
    subparsers.add_parser("preprocess", help="数据预处理（裁剪+CLAHE+增强）")

    # train
    train_parser = subparsers.add_parser("train", help="训练模型")
    train_parser.add_argument("model", type=str,
                              choices=["all", "hog_svm", "lbp_svm", "resnet18",
                                       "resnet18_se", "resnet18_cbam", "mobilenetv3"],
                              help="模型名称")

    # evaluate
    subparsers.add_parser("evaluate", help="评估所有模型并生成报告")

    # report
    subparsers.add_parser("report", help="基于已有结果生成实验报告")

    # visualize
    subparsers.add_parser("visualize", help="生成可视化图表")

    # predict
    predict_parser = subparsers.add_parser("predict", help="单图/批量预测")
    predict_parser.add_argument("--model", type=str, required=True, help="模型名称")
    predict_parser.add_argument("--image", type=str, required=True, help="图像路径")

    # run (一键运行)
    subparsers.add_parser("run", help="一键运行完整流程")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    commands = {
        "preprocess": cmd_preprocess,
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "visualize": cmd_visualize,
        "report": cmd_report,
        "predict": cmd_predict,
        "run": cmd_run,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
