"""
单图/批量推理预测模块
支持所有已训练模型
Author: CleartoCloudy
"""
import os
import sys
import pickle
import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (MODELS_DIR, CLASS_NAMES, IMG_SIZE_TRAD, IMG_SIZE_DL, DEVICE)
from deep_learning.train import build_model
from deep_learning.dataset import get_val_transforms
from traditional.features import extract_hog_features, extract_lbp_features


def load_model(model_name):
    """加载指定模型"""
    model_name = model_name.lower()

    if model_name == "hog_svm":
        path = os.path.join(MODELS_DIR, "hog_svm.pkl")
        with open(path, "rb") as f:
            data = pickle.load(f)
        return {"type": "traditional", "model": data["svm"], "scaler": data["scaler"],
                "feature": "hog"}

    elif model_name == "lbp_svm":
        path = os.path.join(MODELS_DIR, "lbp_svm.pkl")
        with open(path, "rb") as f:
            data = pickle.load(f)
        return {"type": "traditional", "model": data["svm"], "scaler": data["scaler"],
                "feature": "lbp"}

    else:
        # 深度学习模型
        model_type_map = {
            "resnet18": "resnet18",
            "resnet18_se": "resnet18_se",
            "resnet18_cbam": "resnet18_cbam",
            "mobilenetv3": "mobilenetv3",
        }
        model_type = model_type_map.get(model_name, model_name)
        model = build_model(model_type)
        path = os.path.join(MODELS_DIR, f"{model_type}_best.pth")
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        model = model.to(DEVICE)
        model.eval()
        return {"type": "deep_learning", "model": model, "transform": get_val_transforms()}


def predict_single(model_info, image_path, return_probs=False):
    """
    对单张图像进行分类预测

    参数:
        model_info: load_model() 返回的模型信息字典
        image_path: 图像路径
        return_probs: 是否返回各类别概率

    返回: (predicted_class_name, class_id) 或 (name, id, probs)
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")

    if model_info["type"] == "traditional":
        feature_type = model_info["feature"]
        if feature_type == "hog":
            features = extract_hog_features(image, (IMG_SIZE_TRAD, IMG_SIZE_TRAD))
        else:
            features = extract_lbp_features(image, (IMG_SIZE_TRAD, IMG_SIZE_TRAD))

        features = model_info["scaler"].transform([features])
        class_id = model_info["model"].predict(features)[0]

        if return_probs:
            if hasattr(model_info["model"], "predict_proba"):
                probs = model_info["model"].predict_proba(features)[0]
            else:
                # LinearSVC 没有 predict_proba，用 decision_function 转伪概率
                decision = model_info["model"].decision_function(features)[0]
                probs = np.exp(decision) / np.sum(np.exp(decision))
            return CLASS_NAMES[class_id], class_id, probs
        return CLASS_NAMES[class_id], class_id

    else:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb = cv2.resize(image_rgb, (IMG_SIZE_DL, IMG_SIZE_DL))
        tensor = model_info["transform"](image_rgb).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model_info["model"](tensor)
            probs = F.softmax(outputs, dim=1).cpu().numpy()[0]
            class_id = int(torch.argmax(outputs, dim=1).item())

        if return_probs:
            return CLASS_NAMES[class_id], class_id, probs
        return CLASS_NAMES[class_id], class_id


def predict_batch(model_info, image_paths):
    """批量预测，返回结果列表"""
    results = []
    for path in image_paths:
        try:
            name, cid, probs = predict_single(model_info, path, return_probs=True)
            results.append({
                "path": path,
                "class_name": name,
                "class_id": cid,
                "confidence": float(probs[cid]),
                "probabilities": {CLASS_NAMES[i]: round(float(p), 4) for i, p in enumerate(probs)}
            })
        except Exception as e:
            results.append({"path": path, "error": str(e)})
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="PCB缺陷图像分类预测")
    parser.add_argument("--model", type=str, required=True,
                        choices=["hog_svm", "lbp_svm", "resnet18", "resnet18_se",
                                 "resnet18_cbam", "mobilenetv3"],
                        help="模型名称")
    parser.add_argument("--image", type=str, required=True, help="图像路径或目录")
    parser.add_argument("--batch", action="store_true", help="批量预测模式")

    args = parser.parse_args()

    print(f"加载模型: {args.model}")
    model_info = load_model(args.model)
    print(f"模型类型: {model_info['type']}")

    if args.batch or os.path.isdir(args.image):
        # 批量模式
        image_dir = args.image
        image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)
                       if f.endswith((".jpg", ".png", ".bmp"))]
        results = predict_batch(model_info, image_paths)
        correct = sum(1 for r in results if "error" not in r)
        print(f"\n共处理 {len(image_paths)} 张图像, 成功 {correct} 张")
        for r in results[:10]:
            if "error" not in r:
                print(f"  {os.path.basename(r['path'])} -> {r['class_name']} "
                      f"(confidence: {r['confidence']:.4f})")
    else:
        # 单图模式
        name, cid, probs = predict_single(model_info, args.image, return_probs=True)
        print(f"\n图像: {args.image}")
        print(f"预测结果: {name}")
        print(f"置信度: {probs[cid]:.4f}")
        print("\n各类别概率:")
        for i, cls_name in enumerate(CLASS_NAMES):
            print(f"  {cls_name}: {probs[i]:.4f}")


if __name__ == "__main__":
    main()
