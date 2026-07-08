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
  python main.py detect train # 训练检测模型 (Faster R-CNN)
  python main.py detect evaluate # 评估检测模型
  python main.py detect predict --image <path>  # 检测单图
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


def cmd_detect(args):
    """目标检测：训练/评估/预测"""
    if args.action == "train":
        from detection.train import train_detection
        train_detection()
    elif args.action == "evaluate":
        from detection.evaluate import evaluate_detection_model
        evaluate_detection_model()
    elif args.action == "predict":
        from detection.predict import load_detection_model, predict_image
        import cv2
        import json
        from datetime import datetime
        from config import BASE_DIR

        model = load_detection_model()
        threshold = getattr(args, "threshold", 0.3)

        out_dir = os.path.join(BASE_DIR, "results", "reports",
                               f"detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(out_dir, exist_ok=True)

        if os.path.isdir(args.image):
            paths = sorted([os.path.join(args.image, f) for f in os.listdir(args.image)
                           if f.endswith((".jpg", ".png", ".bmp"))])
        else:
            paths = [args.image]

        all_results = []
        for img_path in paths:
            basename = os.path.basename(img_path)
            print(f"\n检测: {basename}")
            try:
                img_result, detections = predict_image(model, img_path, score_thresh=threshold)
                print(f"  发现 {len(detections)} 个缺陷:")
                for det in sorted(detections, key=lambda d: d["score"], reverse=True):
                    x1, y1, x2, y2 = det["box"]
                    print(f"    [{det['class']:20s}] conf={det['score']:.3f}  "
                          f"box=({x1},{y1},{x2},{y2})  size={x2-x1}x{y2-y1}")
                out_path = os.path.join(out_dir, f"det_{os.path.splitext(basename)[0]}.jpg")
                cv2.imwrite(out_path, img_result)
                all_results.append({
                    "image": basename,
                    "num_defects": len(detections),
                    "detections": [{
                        "class": d["class"],
                        "confidence": d["score"],
                        "box": d["box"],
                    } for d in sorted(detections, key=lambda x: x["score"], reverse=True)],
                    "saved_as": os.path.basename(out_path),
                })
                print(f"  已保存: {out_path}")
            except Exception as e:
                print(f"  错误: {e}")
                all_results.append({"image": basename, "error": str(e)})

        # 保存 JSON
        json_path = os.path.join(out_dir, "results.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "model": "Faster R-CNN (ResNet50-FPN)",
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "threshold": threshold,
                "total_images": len(paths),
                "results": all_results,
            }, f, indent=2, ensure_ascii=False)
        print(f"\n结果保存在: {out_dir}")
        print(f"JSON 已保存至: {json_path}")


def cmd_report(args):
    """单独生成实验报告（不重新评估，基于已有结果）"""
    from evaluation.report import generate_report
    generate_report()


def _draw_prediction(image_path, class_name, confidence, probs, save_path):
    """在原图下方扩展区域，写入分类结果"""
    import cv2
    import numpy as np
    from config import CLASS_NAMES

    img = cv2.imread(image_path)
    if img is None:
        return False
    h, w = img.shape[:2]

    # 底部信息栏高度（比例缩放，留足间距）
    pad_h = 120 if w >= 200 else 80
    canvas = np.zeros((h + pad_h, w, 3), dtype=np.uint8)
    canvas[0:h, 0:w] = img
    canvas[h:, :] = (28, 28, 28)

    # 字体缩放
    fs = max(w / 640, 0.4)

    # 置信度颜色
    if confidence > 0.9:
        color = (0, 255, 0)
    elif confidence > 0.7:
        color = (0, 222, 222)
    else:
        color = (0, 100, 255)

    # ---- 顶部横线 ----
    cv2.line(canvas, (0, h), (w, h), color, 2)

    # ---- 主结果 ----
    main_text = f"[ {class_name} ]  {confidence:.1%}"
    cv2.putText(canvas, main_text, (10, h + 28),
                cv2.FONT_HERSHEY_SIMPLEX, fs * 0.7, color, 2)

    # ---- 概率条 ----
    top_idx = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
    n_show = min(6, len(top_idx))
    bar_area_h = pad_h - 36  # 主结果下方可用空间
    row_h = bar_area_h / n_show
    bar_left = 14
    bar_right = w - 14
    bar_area_w = bar_right - bar_left

    for rank, idx in enumerate(top_idx[:n_show]):
        y_center = h + 38 + rank * row_h + row_h * 0.4
        bar_h = row_h * 0.55
        bar_y = int(y_center - bar_h / 2)

        # 背景条
        cv2.rectangle(canvas, (bar_left, bar_y),
                      (bar_right, int(bar_y + bar_h)), (55, 55, 55), -1)

        # 前景条
        bar_w = int(bar_area_w * probs[idx])
        c = color if rank == 0 else (100, 100, 100)
        if bar_w > 0:
            cv2.rectangle(canvas, (bar_left, bar_y),
                          (bar_left + bar_w, int(bar_y + bar_h)), c, -1)

        # 标签文字（类别名 + 百分比）
        label = f"{CLASS_NAMES[idx]}  {probs[idx]:.1%}"
        cv2.putText(canvas, label, (bar_left + 6, int(y_center + bar_h * 0.25)),
                    cv2.FONT_HERSHEY_SIMPLEX, fs * 0.42, (255, 255, 255), 1)

    cv2.imwrite(save_path, canvas)
    return True


def cmd_predict(args):
    """单图/批量预测，结果绘制在图上并保存"""
    import json
    from datetime import datetime
    from deep_learning.predict import load_model, predict_single, predict_batch
    from config import CLASS_NAMES, BASE_DIR

    model_info = load_model(args.model)
    print(f"模型: {args.model}  |  类型: {model_info['type']}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(BASE_DIR, "results", "reports", f"prediction_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    output = {
        "model": args.model,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "predictions": []
    }

    if os.path.isdir(args.image):
        image_paths = sorted([
            os.path.join(args.image, f) for f in os.listdir(args.image)
            if f.endswith((".jpg", ".png", ".bmp"))
        ])
        results = predict_batch(model_info, image_paths)
        output["predictions"] = results

        for r in results:
            basename = os.path.basename(r["path"])
            if "error" in r:
                print(f"  {basename} -> ERROR: {r['error']}")
                continue

            probs = [r["probabilities"][cn] for cn in CLASS_NAMES]
            out_path = os.path.join(out_dir, f"pred_{os.path.splitext(basename)[0]}.jpg")
            _draw_prediction(r["path"], r["class_name"], r["confidence"], probs, out_path)
            print(f"  {basename} -> {r['class_name']} ({r['confidence']:.1%}) => {out_path}")
    else:
        name, cid, probs = predict_single(model_info, args.image, return_probs=True)

        result = {
            "path": args.image,
            "class_name": name,
            "class_id": cid,
            "confidence": float(probs[cid]),
            "probabilities": {CLASS_NAMES[i]: round(float(p), 4) for i, p in enumerate(probs)}
        }
        output["predictions"].append(result)

        basename = os.path.basename(args.image)
        out_path = os.path.join(out_dir, f"pred_{os.path.splitext(basename)[0]}.jpg")
        _draw_prediction(args.image, name, float(probs[cid]), probs, out_path)
        print(f"图像: {args.image}")
        print(f"预测: {name} (置信度: {probs[cid]:.4f})")
        print("各类别概率:")
        for i, cn in enumerate(CLASS_NAMES):
            bar = "█" * int(probs[i] * 40)
            print(f"  {cn:20s}  {probs[i]:.4f}  {bar}")
        print(f"\n标注图已保存至: {out_path}")

    # 保存 JSON
    json_path = os.path.join(out_dir, "results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"JSON 已保存至: {json_path}")


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

    # detect
    detect_parser = subparsers.add_parser("detect", help="目标检测 (Faster R-CNN)")
    detect_sub = detect_parser.add_subparsers(dest="action", help="操作")
    detect_sub.add_parser("train", help="训练检测模型")
    detect_sub.add_parser("evaluate", help="评估检测模型")
    detect_pred = detect_sub.add_parser("predict", help="检测单图/目录")
    detect_pred.add_argument("--image", type=str, required=True, help="图像路径或目录")
    detect_pred.add_argument("--threshold", type=float, default=0.3, help="置信度阈值")

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
        "detect": cmd_detect,
        "predict": cmd_predict,
        "run": cmd_run,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
